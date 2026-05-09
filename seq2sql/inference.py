"""
inference.py
============
Loads trained model and predicts SQL from natural language.
Includes schema linking to map output → exact chinook column names.
Called from app.py for Streamlit integration.
"""

import re
import torch
import sqlite3
import pickle
import sys
from pathlib import Path
from difflib import get_close_matches

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    ENCODER_TYPE, MODEL_DIR, VOCAB_DIR,
    CHINOOK_DB_PATH, CHINOOK_SCHEMA,
    CHINOOK_SCHEMA_CONTEXT, CONFIDENCE_THRESHOLD,
    SOS_IDX, EOS_IDX, PAD_IDX,
    BERT_MODEL_NAME, BERT_MAX_SEQ_LEN,
)
from seq2sql.vocabulary import Vocabulary, tokenize_question, tokenize_schema
from seq2sql.model      import build_model

BEST_MODEL     = MODEL_DIR / f"best_{ENCODER_TYPE}.pt"
FINETUNE_MODEL = MODEL_DIR / f"finetuned_{ENCODER_TYPE}.pt"

# use fine-tuned model if available, else fall back to original
ACTIVE_MODEL = FINETUNE_MODEL if FINETUNE_MODEL.exists() else BEST_MODEL
# ACTIVE_MODEL = BEST_MODEL

# ── Custom unpickler ──────────────────────────────────────────────────────────
# Fixes "Can't get attribute 'Vocabulary' on <module '__main__'>" error
# that occurs when loading pickled vocab files from Streamlit context.

class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)

def _load_vocab(path: Path) -> Vocabulary:
    with open(path, "rb") as f:
        return _VocabUnpickler(f).load()

# ═══════════════════════════════════════════════════════
# Schema linker
# ═══════════════════════════════════════════════════════

class SchemaLinker:
    """
    Maps model-generated SQL (which may use wrong table/column names)
    to correct Chinook schema names using fuzzy matching.
    """

    def __init__(self):
        # flat list of all table names (lowercase)
        self.tables = [t.lower() for t in CHINOOK_SCHEMA.keys()]

        # flat list of all column names (lowercase)
        self.columns = [
            col.lower()
            for cols in CHINOOK_SCHEMA.values()
            for col in cols
        ]

        # table → correct case
        self.table_map = {t.lower(): t for t in CHINOOK_SCHEMA.keys()}

        # column → correct case
        self.col_map = {
            col.lower(): col
            for cols in CHINOOK_SCHEMA.values()
            for col in cols
        }

        # store_1 → chinook_1 name mapping
        # store_1 uses lowercase snake_case, chinook uses PascalCase
        self.store_to_chinook_table = {
            "artists"        : "Artist",
            "albums"         : "Album",
            "customers"      : "Customer",
            "employees"      : "Employee",
            "genres"         : "Genre",
            "invoices"       : "Invoice",
            "invoice_lines"  : "InvoiceLine",
            "media_types"    : "MediaType",
            "playlists"      : "Playlist",
            "playlist_tracks": "PlaylistTrack",
            "tracks"         : "Track",
        }

        self.store_to_chinook_col = {
            "artist_id"         : "ArtistId",
            "album_id"          : "AlbumId",
            "customer_id"       : "CustomerId",
            "employee_id"       : "EmployeeId",
            "genre_id"          : "GenreId",
            "invoice_id"        : "InvoiceId",
            "invoice_line_id"   : "InvoiceLineId",
            "media_type_id"     : "MediaTypeId",
            "playlist_id"       : "PlaylistId",
            "track_id"          : "TrackId",
            "first_name"        : "FirstName",
            "last_name"         : "LastName",
            "birth_date"        : "BirthDate",
            "hire_date"         : "HireDate",
            "reports_to"        : "ReportsTo",
            "support_rep_id"    : "SupportRepId",
            "invoice_date"      : "InvoiceDate",
            "billing_address"   : "BillingAddress",
            "billing_city"      : "BillingCity",
            "billing_state"     : "BillingState",
            "billing_country"   : "BillingCountry",
            "billing_postal_code": "BillingPostalCode",
            "unit_price"        : "UnitPrice",
            "postal_code"       : "PostalCode",
            "media_type_id"     : "MediaTypeId",
        }

    def fix(self, sql: str) -> str:
        """
        Apply all schema corrections to generated SQL.
        """
        # 1. Fix store_1 table names → chinook table names
        for store_name, chinook_name in self.store_to_chinook_table.items():
            sql = re.sub(
                r'\b' + re.escape(store_name) + r'\b',
                chinook_name, sql, flags=re.IGNORECASE
            )

        # 2. Fix store_1 column names → chinook column names
        for store_col, chinook_col in self.store_to_chinook_col.items():
            sql = re.sub(
                r'\b' + re.escape(store_col) + r'\b',
                chinook_col, sql, flags=re.IGNORECASE
            )

        # 3. Fix remaining wrong table names using fuzzy match
        words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', sql)
        for word in set(words):
            if word.lower() in self.table_map:
                correct = self.table_map[word.lower()]
                if word != correct:
                    sql = re.sub(r'\b' + re.escape(word) + r'\b',
                                 correct, sql)
            elif word.lower() in self.col_map:
                correct = self.col_map[word.lower()]
                if word != correct:
                    sql = re.sub(r'\b' + re.escape(word) + r'\b',
                                 correct, sql)

        return sql


# ═══════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════

class NL2SQLEngine:
    """
    Main inference class. Called from app.py.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[NL2SQL] Loading ({ENCODER_TYPE} encoder) ...")

        # load vocabularies using custom unpickler
        # (fixes pickle module path issue when called from Streamlit)
        self.sql_vocab = _load_vocab(VOCAB_DIR / "sql_vocab.pkl")
        self.nl_vocab  = _load_vocab(VOCAB_DIR / "nl_vocab.pkl")

        # load BERT tokenizer if needed
        self.bert_tokenizer = None
        if ENCODER_TYPE == "bert":
            from transformers import BertTokenizerFast
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(
                BERT_MODEL_NAME
            )

        # load model — prefer finetuned_bert.pt if available
        print(f"[NL2SQL] Loading model: {ACTIVE_MODEL.name} ...")
        ckpt = torch.load(ACTIVE_MODEL, map_location=self.device,
                          weights_only=False)
        self.model = build_model(
            sql_vocab_size = len(self.sql_vocab),
            device         = self.device,
            nl_vocab_size  = len(self.nl_vocab),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # schema linker
        self.schema_linker = SchemaLinker()

        print(f"[NL2SQL] Ready on {self.device}")

    def _encode_input(self, question: str,
                      schema_context: str = None):
        """
        Encode question + schema context → tensors for BERT/BiLSTM.

        schema_context: if provided, uses this instead of the static
                        CHINOOK_SCHEMA_CONTEXT. Should come from the
                        TF-IDF schema linker for this specific query
                        so BERT sees only the relevant tables/columns.

        This is the RAG augmentation step — retrieved schema context
        is injected into the encoder input.
        """
        # use query-specific schema context if provided, else fall back to full
        ctx = schema_context if schema_context else CHINOOK_SCHEMA_CONTEXT

        if ENCODER_TYPE == "bert":
            enc = self.bert_tokenizer(
                question,
                ctx,
                max_length     = BERT_MAX_SEQ_LEN,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt"
            )
            src      = enc["input_ids"].to(self.device)
            att_mask = enc["attention_mask"].to(self.device)
        else:
            from seq2sql.vocabulary import tokenize_schema
            q_tok  = tokenize_question(question)
            sc_tok = tokenize_schema(ctx)
            ids    = self.nl_vocab.encode((q_tok + sc_tok)[:256])
            src    = torch.tensor([ids], dtype=torch.long,
                                  device=self.device)
            att_mask = None

        return src, att_mask

    def predict(self, question: str,
                beam_width: int = 5,
                schema_context: str = None) -> dict:
        """
        NL question → SQL string.

        schema_context: query-specific schema from TF-IDF linker.
                        If None, falls back to full static schema.
                        Providing this is the RAG augmentation step.

        Uses beam search (beam_width=5) by default.
        Falls back to greedy if beam search fails.
        """
        src, att_mask = self._encode_input(question, schema_context)

        # ── Beam search (primary) ─────────────────────────────────────
        try:
            beam_ids, attn_weights = self.model.generate_beam(
                src, att_mask,
                max_len    = 100,
                beam_width = beam_width
            )
            decoding_method = f"beam_search (k={beam_width})"
        except Exception:
            # fallback to greedy if beam search errors
            beam_ids, attn_weights = self.model.generate(
                src, att_mask, max_len=100
            )
            decoding_method = "greedy (beam fallback)"

        # ── Greedy (for comparison only) ──────────────────────────────
        try:
            greedy_ids, _ = self.model.generate(
                src, att_mask, max_len=100
            )
            greedy_tokens = self.sql_vocab.decode(greedy_ids)
            if EOS_IDX in greedy_ids:
                greedy_tokens = greedy_tokens[:greedy_ids.index(EOS_IDX)]
            greedy_tokens = [t for t in greedy_tokens
                             if t not in ("<PAD>","<UNK>","<SOS>","<EOS>")]
            greedy_sql = self._tokens_to_sql(greedy_tokens)
            greedy_sql = self.schema_linker.fix(greedy_sql)
        except Exception:
            greedy_sql = ""

        # ── Decode beam result ────────────────────────────────────────
        sql_tokens = self.sql_vocab.decode(beam_ids)
        sql_tokens = [t for t in sql_tokens
                      if t not in ("<PAD>","<UNK>","<SOS>","<EOS>")]

        raw_sql   = self._tokens_to_sql(sql_tokens)
        fixed_sql = self.schema_linker.fix(raw_sql)

        # ── Confidence score ──────────────────────────────────────────
        # Based on UNK fraction — lower UNK = higher confidence
        unk_count  = beam_ids.count(1) if beam_ids else 0
        confidence = 1.0 - unk_count / max(len(beam_ids), 1)

        return {
            "sql"            : fixed_sql,
            "raw_sql"        : raw_sql,
            "greedy_sql"     : greedy_sql,
            "confidence"     : round(confidence, 3),
            "tokens"         : sql_tokens,
            "attn_weights"   : attn_weights,
            "decoding_method": decoding_method,
            "beam_width"     : beam_width,
        }

    def _tokens_to_sql(self, tokens: list) -> str:
        if not tokens:
            return ""
        no_space_before = {")", ",", "."}
        no_space_after  = {"(", "."}
        sql = ""
        for i, tok in enumerate(tokens):
            if i == 0:
                sql += tok
            elif (tok in no_space_before or
                  (i > 0 and tokens[i-1] in no_space_after)):
                sql += tok
            else:
                sql += " " + tok
        return sql.strip()

    def execute(self, sql: str) -> dict:
        """Execute SQL on Chinook DB and return results."""
        try:
            conn   = sqlite3.connect(CHINOOK_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            cols = ([d[0] for d in cursor.description]
                    if cursor.description else [])
            conn.close()
            return {
                "success": True,
                "columns": cols,
                "rows"   : rows,
                "count"  : len(rows),
                "error"  : ""
            }
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows"   : [],
                "count"  : 0,
                "error"  : str(e)
            }

    def query(self, question: str) -> dict:
        """Full pipeline: question → SQL → execute → results."""
        pred   = self.predict(question)
        result = self.execute(pred["sql"])
        return {
            "question"  : question,
            "sql"       : pred["sql"],
            "confidence": pred["confidence"],
            **result
        }


# ═══════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = NL2SQLEngine()

    tests = [
        "how many artists are there",
        "show all customers from usa",
        "what is the total revenue",
        "show top 5 invoices by total",
        "count customers by country",
        "show tracks with their genre names",
        "find top 5 customers by spending",
    ]

    print("\n" + "="*55)
    for q in tests:
        result = engine.query(q)
        print(f"\nQ  : {q}")
        print(f"SQL: {result['sql']}")
        print(f"Conf: {result['confidence']}")
        if result["success"]:
            print(f"Rows: {result['count']}")
            if result["rows"]:
                print(f"Sample: {result['rows'][:2]}")
        else:
            print(f"Error: {result['error']}")
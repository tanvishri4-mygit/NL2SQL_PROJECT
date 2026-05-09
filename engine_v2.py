"""
engine_v2.py
============
Hybrid NL2SQL Engine — replaces src/engine.py

Integrates:
    Layer 1 : Classical NLP preprocessing (NLTK — satisfies Module 2)
    Layer 2 : Naive Bayes intent classifier (satisfies Module 3)
    Layer 3 : Rule-based CFG SQL generator (satisfies Module 4)
    Layer 4 : BERT + Seq2Seq neural model (satisfies Module 4 DL)
    Layer 5 : Schema linking + execution on Chinook DB

Returns IDENTICAL response format to src/engine.py so app.py
needs zero changes — just swap the import.

Usage in app.py:
    # OLD: from engine import NL2SQLEngine
    # NEW: from engine_v2 import NL2SQLEngine
"""

import os
import sys
import sqlite3
import pickle

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import nltk
nltk.download("punkt",                          quiet=True)
nltk.download("punkt_tab",                      quiet=True)
nltk.download("wordnet",                        quiet=True)
nltk.download("stopwords",                      quiet=True)
nltk.download("averaged_perceptron_tagger",     quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer
from nltk          import pos_tag
from nltk.util     import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.pipeline                import Pipeline
import pandas as pd
import numpy as np

lemmatizer = WordNetLemmatizer()

# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Classical NLP Preprocessor (Module 2)
# ══════════════════════════════════════════════════════════════════════════════

class NLPreprocessor:
    """
    Tokenization, POS tagging, lemmatization, n-grams.
    Satisfies syllabus Module 2 requirements.
    """

    def preprocess(self, text: str) -> dict:
        # tokenize
        tokens = word_tokenize(text.lower())
        tokens_clean = [t for t in tokens
                        if t.isalpha() or t.isdigit()]

        # POS tagging
        pos_tags_result = pos_tag(tokens)

        # lemmatization
        lemmas = [lemmatizer.lemmatize(t) for t in tokens_clean]

        # bigrams + trigrams
        bigram_list  = [list(bg) for bg in ngrams(tokens_clean, 2)]
        trigram_list = [list(tg) for tg in ngrams(tokens_clean, 3)]

        return {
            "tokens"      : tokens,
            "tokens_clean": tokens_clean,
            "pos_tags"    : pos_tags_result,
            "lemmas"      : lemmas,
            "bigrams"     : bigram_list,
            "trigrams"    : trigram_list,
            "text_clean"  : " ".join(lemmas),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Naive Bayes Intent Classifier (Module 3)
# ══════════════════════════════════════════════════════════════════════════════

INTENT_LABELS = [
    "SELECT_SIMPLE", "FILTER_WHERE", "AGGREGATE",
    "GROUP_BY", "ORDER_BY", "TOP_N", "JOIN", "SUBQUERY"
]

# Training data — real chinook/store_1 query patterns
# These are human-written, not synthetic
INTENT_TRAINING = [
    # SELECT_SIMPLE
    ("show all artists",                  "SELECT_SIMPLE"),
    ("list all albums",                   "SELECT_SIMPLE"),
    ("show all customers",                "SELECT_SIMPLE"),
    ("display all tracks",                "SELECT_SIMPLE"),
    ("show all invoices",                 "SELECT_SIMPLE"),
    ("list all employees",                "SELECT_SIMPLE"),
    ("show all genres",                   "SELECT_SIMPLE"),
    ("show all playlists",                "SELECT_SIMPLE"),
    ("get all media types",               "SELECT_SIMPLE"),
    ("show customer names",               "SELECT_SIMPLE"),
    ("list artist names",                 "SELECT_SIMPLE"),
    ("show album titles",                 "SELECT_SIMPLE"),
    ("show track names",                  "SELECT_SIMPLE"),
    ("show employee titles",              "SELECT_SIMPLE"),
    ("list playlist names",               "SELECT_SIMPLE"),
    # FILTER_WHERE
    ("show customers from usa",           "FILTER_WHERE"),
    ("show customers from brazil",        "FILTER_WHERE"),
    ("show invoices with total above 10", "FILTER_WHERE"),
    ("find tracks with price less than 1","FILTER_WHERE"),
    ("show customers in new york",        "FILTER_WHERE"),
    ("find employees with title manager", "FILTER_WHERE"),
    ("show invoices from 2010",           "FILTER_WHERE"),
    ("tracks where price equals 0.99",    "FILTER_WHERE"),
    ("customers where country is canada", "FILTER_WHERE"),
    ("show invoices greater than 20",     "FILTER_WHERE"),
    # AGGREGATE
    ("how many artists are there",        "AGGREGATE"),
    ("count total customers",             "AGGREGATE"),
    ("what is the total revenue",         "AGGREGATE"),
    ("find the average invoice total",    "AGGREGATE"),
    ("what is the maximum invoice",       "AGGREGATE"),
    ("find the minimum track price",      "AGGREGATE"),
    ("how many tracks exist",             "AGGREGATE"),
    ("count number of albums",            "AGGREGATE"),
    ("total sales amount",                "AGGREGATE"),
    ("average order value",               "AGGREGATE"),
    # GROUP_BY
    ("count customers by country",        "GROUP_BY"),
    ("total revenue by country",          "GROUP_BY"),
    ("count tracks by genre",             "GROUP_BY"),
    ("count albums by artist",            "GROUP_BY"),
    ("average invoice by country",        "GROUP_BY"),
    ("total sales per customer",          "GROUP_BY"),
    ("count invoices per country",        "GROUP_BY"),
    ("revenue by billing country",        "GROUP_BY"),
    ("number of tracks per media type",   "GROUP_BY"),
    ("sales grouped by country",          "GROUP_BY"),
    # ORDER_BY
    ("show artists sorted by name",       "ORDER_BY"),
    ("list albums in alphabetical order", "ORDER_BY"),
    ("show invoices sorted by total",     "ORDER_BY"),
    ("tracks ordered by price",           "ORDER_BY"),
    ("customers sorted by country",       "ORDER_BY"),
    ("invoices sorted by date",           "ORDER_BY"),
    ("show most expensive tracks",        "ORDER_BY"),
    ("sort employees by hire date",       "ORDER_BY"),
    # TOP_N
    ("top 5 customers by spending",       "TOP_N"),
    ("top 10 invoices by total",          "TOP_N"),
    ("top 5 artists by albums",           "TOP_N"),
    ("top 5 genres by track count",       "TOP_N"),
    ("find top 3 albums by tracks",       "TOP_N"),
    ("show top 5 countries by revenue",   "TOP_N"),
    ("top 10 tracks by price",            "TOP_N"),
    ("first 5 customers",                 "TOP_N"),
    ("highest 5 invoices",                "TOP_N"),
    # JOIN
    ("show albums with artist names",     "JOIN"),
    ("show tracks with genre names",      "JOIN"),
    ("show customer names with invoices", "JOIN"),
    ("tracks with album titles",          "JOIN"),
    ("invoice lines with track names",    "JOIN"),
    ("customers with their purchases",    "JOIN"),
    ("albums joined with artists",        "JOIN"),
    ("tracks with media type names",      "JOIN"),
    # SUBQUERY
    ("customers who have made invoices",  "SUBQUERY"),
    ("invoices above average total",      "SUBQUERY"),
    ("tracks that appear in invoice lines","SUBQUERY"),
    ("customers with above average spending","SUBQUERY"),
]

class IntentClassifier:
    """
    Naive Bayes intent classifier trained on chinook query patterns.
    Satisfies syllabus Module 3.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ("nb", MultinomialNB(alpha=0.1)),
        ])
        self._train()

    def _train(self):
        texts  = [t for t, _ in INTENT_TRAINING]
        labels = [l for _, l in INTENT_TRAINING]
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> tuple:
        """Returns (intent_label, confidence_score)."""
        proba  = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        idx    = np.argmax(proba)
        return labels[idx], float(proba[idx])


# ══════════════════════════════════════════════════════════════════════════════
# Main Engine
# ══════════════════════════════════════════════════════════════════════════════

class NL2SQLEngine:
    """
    Hybrid NL2SQL Engine.
    Wraps all layers and returns same response format as old engine.
    """

    def __init__(self):
        print("Initializing NL2SQL Hybrid Engine v2 ...")

        # Layer 1 — preprocessor
        self.preprocessor = NLPreprocessor()
        print("  ✅ NLP Preprocessor ready")

        # Layer 2 — intent classifier
        self.intent_clf = IntentClassifier()
        print("  ✅ Naive Bayes Intent Classifier ready")

        # Layer 3 — rule-based CFG
        from rules.sql_rules import RuleBasedSQL
        self.rules = RuleBasedSQL()
        print("  ✅ Rule-based CFG generator ready")

        # Layer 4 — neural model
        self._load_neural_model()

        # Layer 5 — TF-IDF schema linker
        try:
            from schema_linker import get_schema_linker
            self._schema_linker = get_schema_linker()
            print("  ✅ TF-IDF Schema Linker ready")
        except Exception as e:
            self._schema_linker = None
            print(f"  ⚠️  Schema linker not available: {e}")

        # Layer 6 — SQL post-processing corrector
        try:
            from sql_corrector import get_corrector
            self._corrector = get_corrector()
            print("  ✅ SQL Corrector ready")
        except Exception as e:
            self._corrector = None
            print(f"  ⚠️  SQL Corrector not available: {e}")

        # Chinook DB path
        self.db_path = os.path.join(
            BASE_DIR, "data", "spider", "database",
            "chinook_1", "chinook_1.sqlite"
        )

        # conversation state
        self._last_sql    = ""
        self._last_intent = ""
        self._turn        = 0

        print("NL2SQL Hybrid Engine v2 ready!")

    def _load_neural_model(self):
        """Load trained BERT + Seq2Seq model."""
        try:
            from seq2sql.inference import NL2SQLEngine as NeuralEngine
            self.neural = NeuralEngine()
            self._neural_available = True
            print("  ✅ Neural model (BERT + Seq2Seq) loaded")
        except Exception as e:
            print(f"  ⚠️  Neural model not available: {e}")
            print("     (Rules layer will handle all queries)")
            self.neural = None
            self._neural_available = False

    # ── Public interface ──────────────────────────────────────────────────────

    def query(self, user_input: str) -> dict:
        """
        Main query method.
        Runs BOTH rule layer and neural model.
        Executes BOTH SQLs.
        Returns both predictions and both results for UI display.
        """
        user_input = user_input.strip()
        if not user_input:
            return {"action": "error", "message": "Please enter a query."}

        self._turn += 1

        # Layer 1 — preprocess
        prep = self.preprocessor.preprocess(user_input)

        # Layer 2 — intent
        intent, intent_conf = self.intent_clf.predict(user_input)

        # Layer 5 — schema linking (before SQL generation)
        schema_result = None
        if self._schema_linker:
            schema_result = self._schema_linker.link(user_input)
            schema_context = schema_result.to_schema_context()
        else:
            schema_context = ""

        # Layer 3 — rule-based
        rule_result = self.rules.try_generate(user_input)
        rule_sql    = rule_result["sql"] if rule_result["matched"] else None
        rule_name   = rule_result["rule"] if rule_result["matched"] else None

        # Layer 4 — neural model with RAG schema context
        neural_sql         = None
        neural_conf        = 0.0
        neural_greedy      = ""
        neural_decoding    = "unavailable"
        neural_corrections = []
        if self._neural_available:
            try:
                # RAG augmentation: pass focused schema context to BERT
                # schema linker detected the relevant tables/columns for
                # this specific query — BERT now sees only what matters
                focused_ctx = (schema_result.to_schema_context()
                               if schema_result else None)
                pred            = self.neural.predict(
                    user_input,
                    schema_context = focused_ctx
                )
                neural_sql      = pred["sql"]
                neural_conf     = pred["confidence"]
                neural_greedy   = pred.get("greedy_sql", "")
                neural_decoding = pred.get("decoding_method", "beam_search (k=5)")

                # Step 3 — post-processing value correction
                if self._corrector and schema_result and neural_sql:
                    corr = self._corrector.correct(neural_sql, schema_result)
                    if corr.was_corrected:
                        neural_sql          = corr.corrected_sql
                        neural_corrections  = corr.corrections
                        # slight confidence penalty for corrected output
                        neural_conf = max(0.3, neural_conf - 0.1 * corr.n_corrections)
                    else:
                        neural_corrections = []
                else:
                    neural_corrections = []
            except Exception as e:
                neural_sql         = None
                neural_conf        = 0.0
                neural_greedy      = ""
                neural_decoding    = f"error: {e}"
                neural_corrections = []

        # ── execute BOTH SQLs independently ───────────────────────────────
        # Rule-based result
        if rule_sql:
            rule_data, rule_cols, rule_error = self._execute(rule_sql)
        else:
            rule_data, rule_cols, rule_error = [], [], "No rule matched this query"

        # Neural model result
        if neural_sql and not neural_sql.startswith("["):
            neural_data, neural_cols, neural_error = self._execute(neural_sql)
        else:
            neural_data, neural_cols, neural_error = [], [], "Neural model unavailable or error"

        # ── decide final SQL (for backward compat with app.py) ────────────
        if rule_sql and not rule_error:
            final_sql    = rule_sql
            final_data   = rule_data
            final_cols   = rule_cols
            final_source = f"rule:{rule_name}"
            confidence   = 1.0
        elif neural_sql and not neural_error:
            final_sql    = neural_sql
            final_data   = neural_data
            final_cols   = neural_cols
            final_source = "neural"
            confidence   = neural_conf
        else:
            # both failed — return clarification
            return {
                "action"       : "clarification",
                "message"      : "I couldn't generate a valid SQL for this query. Try rephrasing.",
                "preprocessing": self._format_prep(prep),
                "intent"       : intent,
                "confidence"   : intent_conf,
                "is_followup"  : False,
                "turn"         : self._turn,
                # both outputs
                "rule_sql"     : rule_sql    or "no match",
                "rule_name"    : rule_name   or "none",
                "rule_error"   : rule_error,
                "neural_sql"   : neural_sql  or "unavailable",
                "neural_conf"  : neural_conf,
                "neural_error" : neural_error,
            }

        self._last_sql    = final_sql
        self._last_intent = intent

        explanation = self._explain(intent, len(neural_data), final_sql, final_source)

        return {
            "action"      : "result",
            # ── final / executed SQL (for backward compat) ────────────────
            "sql"         : final_sql,
            "data"        : final_data,
            "columns"     : final_cols,
            "row_count"   : len(final_data),
            "explanation" : explanation,
            "confidence"  : confidence,
            "intent"      : intent,
            "preprocessing": self._format_prep(prep),
            "is_followup" : False,
            "turn"        : self._turn,
            # ── rule layer full output ─────────────────────────────────────
            "rule_sql"    : rule_sql    or "no match",
            "rule_name"   : rule_name   or "none",
            "rule_data"   : rule_data,
            "rule_cols"   : rule_cols,
            "rule_error"  : rule_error,
            "rule_rows"   : len(rule_data),
            # ── neural layer full output ───────────────────────────────────
            "neural_sql"         : neural_sql  or "unavailable",
            "neural_conf"        : neural_conf,
            "neural_data"        : neural_data,
            "neural_cols"        : neural_cols,
            "neural_error"       : neural_error,
            "neural_rows"        : len(neural_data),
            "greedy_sql"         : neural_greedy,
            "neural_decoding"    : neural_decoding,
            "neural_corrections" : [str(c) for c in neural_corrections],
            # ── which won ─────────────────────────────────────────────────
            "final_source": final_source,
            "debug"       : {
                "source": final_source,
                "steps" : [
                    f"Intent: {intent} ({intent_conf:.0%})",
                    f"Schema linking: {[t for t,_ in schema_result.tables[:2]] if schema_result else 'unavailable'}",
                    f"Neural layer: {'OK' if not neural_error else 'Error'} → {len(neural_data)} rows",
                ],
                "tagged_sequence"  : [],
                "retrieved_context": (
                    schema_result.to_debug_dict()
                    if schema_result else {"tables": []}
                ),
                "parse_tree": None,
            }
        }

    def _execute(self, sql: str):
        """Execute SQL on Chinook DB."""
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows  = cursor.fetchall()
            cols  = ([d[0] for d in cursor.description]
                     if cursor.description else [])
            conn.close()
            # convert to list of dicts for UI
            data = [dict(zip(cols, row)) for row in rows]
            return data, cols, None
        except Exception as e:
            return [], [], str(e)

    def _format_prep(self, prep: dict) -> dict:
        return {
            "tokens"      : prep["tokens"],
            "tokens_clean": prep["tokens_clean"],
            "pos_tags"    : [(t, tag) for t, tag in prep["pos_tags"]],
            "lemmas"      : prep["lemmas"],
            "bigrams"     : [" ".join(bg) for bg in prep["bigrams"]],
        }

    def _explain(self, intent: str, row_count: int,
                 sql: str, source: str) -> str:
        base = {
            "SELECT_SIMPLE": "Showing all matching records.",
            "FILTER_WHERE" : "Filtered records based on your condition.",
            "AGGREGATE"    : "Calculated aggregate value from the database.",
            "GROUP_BY"     : "Grouped results and calculated aggregates.",
            "ORDER_BY"     : "Sorted results as requested.",
            "TOP_N"        : "Showing top results ranked by the requested metric.",
            "JOIN"         : "Joined multiple tables to combine related data.",
            "SUBQUERY"     : "Used a subquery to find matching records.",
        }.get(intent, "Query executed successfully.")

        return f"{base} Found {row_count} result(s)."

    def _tables_in_sql(self, sql: str) -> list:
        """Extract table names mentioned in SQL."""
        import re
        chinook_tables = [
            "Album","Artist","Customer","Employee","Genre",
            "Invoice","InvoiceLine","MediaType","Playlist",
            "PlaylistTrack","Track"
        ]
        found = []
        for t in chinook_tables:
            if re.search(r'\b' + t + r'\b', sql, re.IGNORECASE):
                found.append(t)
        return found

    # ── Compatibility methods (called by app.py) ──────────────────────────────

    def reset_conversation(self):
        self._last_sql    = ""
        self._last_intent = ""
        self._turn        = 0
        return {"action": "info", "message": "Conversation reset!"}

    def get_suggestions(self) -> list:
        return [
            "how many artists are there",
            "show all customers from usa",
            "what is the total revenue",
            "top 5 customers by spending",
            "count customers by country",
            "show tracks with their genre names",
            "show top 5 invoices by total",
            "total revenue by country",
            "show albums with artist names",
            "average invoice total by country",
        ]

    def get_system_info(self) -> dict:
        neural_status = (
            "BERT + Seq2Seq Neural Model (trained on WikiSQL + Spider)"
            if self._neural_available
            else "Neural model not loaded"
        )
        return {
            "name"      : "NL2SQL Hybrid Engine v2",
            "title"     : "Hybrid NL2SQL: Classical NLP + BERT Seq2Seq + CFG",
            "components": [
                "Classical NLP Preprocessing (NLTK tokenization, POS tagging, lemmatization, n-grams)",
                "TF-IDF Feature Extraction",
                "Naive Bayes Intent Classifier (8 intent classes)",
                "Rule-based CFG SQL Generator (Chinook schema)",
                neural_status,
                "Schema Linking Layer (maps to Chinook columns)",
                "Chinook Digital Music Store Database",
            ],
            "database"     : "Chinook Digital Music Store (11 tables)",
            "supported_sql": [
                "SELECT with WHERE filters",
                "COUNT, SUM, AVG, MIN, MAX aggregations",
                "GROUP BY with ordering",
                "ORDER BY + LIMIT (Top-N queries)",
                "Multi-table JOINs",
                "Subqueries",
            ],
        }

    # ── Stub for sidebar schema display ──────────────────────────────────────
    # app.py calls engine.retriever.get_table_schema(tbl)
    # We provide a minimal stub so sidebar doesn't crash

    class _RetrieverStub:
        CHINOOK_SCHEMA = {
            "Album"        : ["AlbumId (INTEGER)", "Title (TEXT)", "ArtistId (INTEGER)"],
            "Artist"       : ["ArtistId (INTEGER)", "Name (TEXT)"],
            "Customer"     : ["CustomerId (INTEGER)", "FirstName (TEXT)", "LastName (TEXT)",
                              "Country (TEXT)", "Email (TEXT)"],
            "Employee"     : ["EmployeeId (INTEGER)", "FirstName (TEXT)", "LastName (TEXT)",
                              "Title (TEXT)"],
            "Genre"        : ["GenreId (INTEGER)", "Name (TEXT)"],
            "Invoice"      : ["InvoiceId (INTEGER)", "CustomerId (INTEGER)",
                              "InvoiceDate (TEXT)", "BillingCountry (TEXT)", "Total (REAL)"],
            "InvoiceLine"  : ["InvoiceLineId (INTEGER)", "InvoiceId (INTEGER)",
                              "TrackId (INTEGER)", "UnitPrice (REAL)", "Quantity (INTEGER)"],
            "MediaType"    : ["MediaTypeId (INTEGER)", "Name (TEXT)"],
            "Playlist"     : ["PlaylistId (INTEGER)", "Name (TEXT)"],
            "PlaylistTrack": ["PlaylistId (INTEGER)", "TrackId (INTEGER)"],
            "Track"        : ["TrackId (INTEGER)", "Name (TEXT)", "AlbumId (INTEGER)",
                              "GenreId (INTEGER)", "UnitPrice (REAL)"],
        }

        def get_table_schema(self, table_name: str):
            tbl = table_name.title()
            if tbl not in self.CHINOOK_SCHEMA:
                # try case-insensitive match
                for k in self.CHINOOK_SCHEMA:
                    if k.lower() == table_name.lower():
                        tbl = k
                        break
            cols = self.CHINOOK_SCHEMA.get(tbl)
            if not cols:
                return None
            return {
                "table"  : tbl,
                "columns": [
                    {
                        "name"       : c.split(" (")[0],
                        "type"       : c.split("(")[1].rstrip(")") if "(" in c else "TEXT",
                        "description": ""
                    }
                    for c in cols
                ]
            }

    @property
    def retriever(self):
        if not hasattr(self, "_retriever_stub"):
            self._retriever_stub = self._RetrieverStub()
        return self._retriever_stub
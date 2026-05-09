"""
dataset.py
==========
PyTorch Dataset + DataLoader.
Handles both BERT and BiLSTM input formats automatically
based on config.ENCODER_TYPE.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
from pathlib import Path

# ensure project root is in path regardless of where script is run from
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    ENCODER_TYPE, PROCESSED_DIR,
    BERT_MODEL_NAME, BERT_MAX_SEQ_LEN,
    MAX_SRC_LEN, MAX_TGT_LEN,
    PAD_IDX, SOS_IDX, EOS_IDX,
    BATCH_SIZE
)
from seq2sql.vocabulary import (
    Vocabulary, tokenize_question,
    tokenize_sql, tokenize_schema
)

# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

class NL2SQLDataset(Dataset):
    """
    Returns (src, tgt) pairs where:
        BERT mode  : src = BERT input_ids (question + [SEP] + schema)
        BiLSTM mode: src = NL vocab indices (question + schema tokens)
        tgt        = SQL vocab indices with SOS/EOS
    """

    def __init__(
        self,
        csv_path : Path,
        sql_vocab: Vocabulary,
        nl_vocab : Vocabulary = None,   # only used in bilstm mode
        bert_tokenizer = None,          # only used in bert mode
    ):
        self.sql_vocab      = sql_vocab
        self.nl_vocab       = nl_vocab
        self.bert_tokenizer = bert_tokenizer
        self.encoder_type   = ENCODER_TYPE

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["question", "sql"])
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        q      = str(row["question"])
        sql    = str(row["sql"])
        schema = str(row["schema_context"]) if "schema_context" in row.index and pd.notna(row["schema_context"]) else ""

        # ── target (same for both modes) ──────────────
        sql_tokens = tokenize_sql(sql)[:MAX_TGT_LEN - 2]
        tgt_ids    = ([SOS_IDX]
                      + self.sql_vocab.encode(sql_tokens)
                      + [EOS_IDX])
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        # ── source ────────────────────────────────────
        if self.encoder_type == "bert":
            # BERT: encode question + schema together
            # Format: [CLS] question [SEP] schema [SEP]
            encoding = self.bert_tokenizer(
                q,
                schema,
                max_length     = BERT_MAX_SEQ_LEN,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt"
            )
            src_tensor = encoding["input_ids"].squeeze(0)      # (seq_len,)
            att_mask   = encoding["attention_mask"].squeeze(0) # (seq_len,)
            return src_tensor, att_mask, tgt_tensor

        else:
            # BiLSTM: question + schema tokens → NL vocab indices
            q_tok  = tokenize_question(q)
            sc_tok = tokenize_schema(schema)
            src_tokens = (q_tok + sc_tok)[:MAX_SRC_LEN]
            src_ids    = self.nl_vocab.encode(src_tokens)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            return src_tensor, None, tgt_tensor


# ═══════════════════════════════════════════════════════
# Collate functions
# ═══════════════════════════════════════════════════════

def collate_bert(batch):
    src_ids, att_masks, tgts = zip(*batch)
    src_ids   = torch.stack(src_ids)    # (B, seq_len) — already padded by BERT tokenizer
    att_masks = torch.stack(att_masks)  # (B, seq_len)
    tgts      = pad_sequence(tgts, batch_first=True,
                             padding_value=PAD_IDX)
    return src_ids, att_masks, tgts

def collate_bilstm(batch):
    src_ids, _, tgts = zip(*batch)
    src_padded = pad_sequence(src_ids, batch_first=True,
                              padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgts,    batch_first=True,
                              padding_value=PAD_IDX)
    # return None for att_mask to keep same interface
    return src_padded, None, tgt_padded


# ═══════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════

def get_dataloaders(
    sql_vocab     : Vocabulary,
    nl_vocab      : Vocabulary = None,
    bert_tokenizer = None,
    batch_size    : int = BATCH_SIZE,
):
    collate_fn = (collate_bert
                  if ENCODER_TYPE == "bert"
                  else collate_bilstm)

    def make(split):
        ds = NL2SQLDataset(
            csv_path       = PROCESSED_DIR / f"{split}.csv",
            sql_vocab      = sql_vocab,
            nl_vocab       = nl_vocab,
            bert_tokenizer = bert_tokenizer,
        )
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            collate_fn  = collate_fn,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

    train_loader = make("train")
    val_loader   = make("val")
    test_loader  = make("test")

    print(f"  DataLoaders ({ENCODER_TYPE} mode):")
    print(f"    Train: {len(train_loader)} batches")
    print(f"    Val  : {len(val_loader)} batches")
    print(f"    Test : {len(test_loader)} batches")

    return train_loader, val_loader, test_loader

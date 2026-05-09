"""
finetune.py
===========
Fine-tunes best_bert.pt on Chinook-specific training pairs.

Key design decisions:
- NEVER overwrites best_bert.pt — loads it as starting point only
- Saves fine-tuned model to models/finetuned_bert.pt
- Saves resume checkpoint to models/resume_finetune.pt
- Saves training log to models/log_finetune.csv
- Trains ONLY on chinook_pairs.csv with 10x oversampling
- Uses LR = 1e-5 (10x smaller than original 1e-4) so model
  retains WikiSQL SQL syntax knowledge while learning Chinook patterns
- Early stopping with patience=5

Why oversampling:
  481 Chinook pairs vs 64K original pairs = ratio 1:133
  Without oversampling, Chinook pairs appear once per epoch
  With 10x oversampling, they appear 10 times per epoch
  This gives the model enough exposure to fix Chinook patterns

Run from project root on Abhishek's laptop:
    conda activate nl2sql
    python seq2sql/finetune.py

Expected time: 30-60 minutes on RTX 4070
Expected result: finetuned_bert.pt with significantly better
                 Chinook-specific SQL generation
"""

import math
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

# ── path setup ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    ENCODER_TYPE, MODEL_DIR, VOCAB_DIR, DATA_DIR,
    BERT_MAX_SEQ_LEN, BERT_MODEL_NAME,
    CLIP_GRAD, PAD_IDX, SOS_IDX, EOS_IDX,
    CHINOOK_SCHEMA_CONTEXT,
)
from seq2sql.vocabulary import Vocabulary, tokenize_question, tokenize_sql
from seq2sql.model      import build_model

# ── fine-tune specific config ─────────────────────────────────────────
FINETUNE_LR        = 1e-5       # 10x smaller than original LR_BERT=1e-4
FINETUNE_EPOCHS    = 20         # max epochs — early stopping will trigger
FINETUNE_BATCH     = 16         # smaller batch for Chinook data
FINETUNE_PATIENCE  = 5          # stop if no improvement for 5 epochs
OVERSAMPLE_FACTOR  = 10         # repeat chinook pairs 10x per epoch
TEACHER_FORCING    = 0.5        # start at 50% teacher forcing
TF_DECAY           = 0.05       # decay per epoch
TF_MIN             = 0.1        # minimum teacher forcing

SOURCE_MODEL       = MODEL_DIR / "best_bert.pt"          # never touched
FINETUNE_MODEL     = MODEL_DIR / "finetuned_bert.pt"     # new best
RESUME_MODEL       = MODEL_DIR / "resume_finetune.pt"    # resume checkpoint
LOG_FILE           = MODEL_DIR / "log_finetune.csv"
CHINOOK_CSV        = DATA_DIR  / "chinook_pairs.csv"


# ═══════════════════════════════════════════════════════
# Chinook-specific Dataset
# ═══════════════════════════════════════════════════════

class ChinookDataset(Dataset):
    """
    Dataset that reads directly from chinook_pairs.csv.
    Uses BERT tokenizer to encode question + schema context.
    SQL is encoded using the existing sql_vocab.

    The schema context fed to BERT is CHINOOK_SCHEMA_CONTEXT
    from config — same context used during inference.
    """

    def __init__(self, csv_path: Path, sql_vocab: Vocabulary,
                 bert_tokenizer, oversample: int = 1):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["question", "sql"])
        self.data           = df.reset_index(drop=True)
        self.sql_vocab      = sql_vocab
        self.bert_tokenizer = bert_tokenizer
        self.oversample     = oversample
        self._len           = len(self.data) * oversample
        print(f"  ChinookDataset: {len(self.data)} pairs "
              f"× {oversample}x oversampling = {self._len} samples")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # oversampling — wrap index
        real_idx = idx % len(self.data)
        row      = self.data.iloc[real_idx]

        question = str(row["question"])
        sql      = str(row["sql"])

        # ── BERT encoding (question + schema context) ──
        enc = self.bert_tokenizer(
            question,
            CHINOOK_SCHEMA_CONTEXT,
            max_length     = BERT_MAX_SEQ_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        src_ids  = enc["input_ids"].squeeze(0)      # (256,)
        att_mask = enc["attention_mask"].squeeze(0)  # (256,)

        # ── SQL encoding ──────────────────────────────
        sql_tokens = tokenize_sql(sql)
        sql_ids    = self.sql_vocab.encode(sql_tokens)

        # add SOS and EOS
        tgt = torch.tensor(
            [SOS_IDX] + sql_ids[:126] + [EOS_IDX],
            dtype=torch.long
        )

        return src_ids, att_mask, tgt


def collate_chinook(batch):
    """Collate for BERT mode — src is already padded to BERT_MAX_SEQ_LEN."""
    src_ids, att_masks, tgts = zip(*batch)

    src_padded  = torch.stack(src_ids,  dim=0)   # (B, 256)
    mask_padded = torch.stack(att_masks, dim=0)  # (B, 256)
    tgt_padded  = pad_sequence(tgts, batch_first=True,
                               padding_value=PAD_IDX)
    return src_padded, mask_padded, tgt_padded


# ═══════════════════════════════════════════════════════
# Train / eval one epoch (same as train.py)
# ═══════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, criterion,
              device, train: bool, tf_ratio: float = 0.0):
    from tqdm import tqdm
    model.train() if train else model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    mode = "Train" if train else "Val  "
    pbar = tqdm(loader, desc=f"  {mode}", leave=False,
                ncols=90, unit="batch")

    for src, att_mask, tgt in pbar:
        src      = src.to(device)
        tgt      = tgt.to(device)
        att_mask = att_mask.to(device) if att_mask is not None else None

        if train:
            optimizer.zero_grad()
            output = model(src, att_mask, tgt,
                           teacher_forcing_ratio=tf_ratio)
        else:
            with torch.no_grad():
                output = model(src, att_mask, tgt,
                               teacher_forcing_ratio=0.0)

        B, T, V = output.shape
        loss = criterion(
            output.reshape(B * T, V),
            tgt[:, 1:].reshape(B * T)
        )

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds   = output.reshape(B * T, V).argmax(dim=1)
            targets = tgt[:, 1:].reshape(B * T)
            mask    = targets != PAD_IDX
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_tokens  += mask.sum().item()

        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.3f}"})

    avg_loss  = total_loss / len(loader)
    token_acc = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    return avg_loss, token_acc


@torch.no_grad()
def show_chinook_samples(model, sql_vocab, bert_tokenizer, device):
    """
    Run a few fixed Chinook test queries through the fine-tuned model
    to visually verify improvement. Shown every 5 epochs.
    """
    model.eval()
    test_pairs = [
        ("show all artists",                    "SELECT * FROM Artist"),
        ("show customers from usa",             "SELECT * FROM Customer WHERE Country = 'USA'"),
        ("how many artists are there",          "SELECT COUNT(*) FROM Artist"),
        ("total revenue by country",            "SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry"),
        ("top 5 artists by album count",        "SELECT Artist.Name, COUNT(Album.AlbumId) FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId GROUP BY Artist.ArtistId ORDER BY COUNT(Album.AlbumId) DESC LIMIT 5"),
    ]

    print(f"\n  {'─'*65}")
    print(f"  Chinook sample predictions:")
    print(f"  {'─'*65}")

    for question, expected in test_pairs:
        enc = bert_tokenizer(
            question,
            CHINOOK_SCHEMA_CONTEXT,
            max_length     = BERT_MAX_SEQ_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        src      = enc["input_ids"].to(device)
        att_mask = enc["attention_mask"].to(device)

        # beam search
        token_ids, _ = model.generate_beam(src, att_mask,
                                            max_len=100, beam_width=5)
        sql_tokens   = sql_vocab.decode(token_ids)
        sql_tokens   = [t for t in sql_tokens
                        if t not in ("<PAD>","<UNK>","<SOS>","<EOS>")]
        pred_sql     = " ".join(sql_tokens)

        match = "[PASS]" if expected.upper() in pred_sql.upper() else "[FAIL]"
        print(f"  Q: {question}")
        print(f"  P: {pred_sql[:90]}")
        print(f"  E: {expected[:90]}  {match}")
        print()

    print(f"  {'─'*65}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  NL2SQL Fine-tuning — Chinook-specific training")
    print("="*60)

    # ── Device ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : "
              f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # ── Verify source model exists ────────────────────
    if not SOURCE_MODEL.exists():
        print(f"\n❌ ERROR: {SOURCE_MODEL} not found.")
        print("  Fine-tuning requires best_bert.pt to exist.")
        return

    if not CHINOOK_CSV.exists():
        print(f"\n❌ ERROR: {CHINOOK_CSV} not found.")
        print("  Run: python checks/generate_chinook_training_data.py")
        return

    print(f"\n  Source model : {SOURCE_MODEL}")
    print(f"  Output model : {FINETUNE_MODEL}")
    print(f"  Training data: {CHINOOK_CSV}")

    # ── Load vocabularies ─────────────────────────────
    print("\n[1] Loading vocabularies ...")
    sql_vocab = Vocabulary.load(VOCAB_DIR / "sql_vocab.pkl")
    nl_vocab  = Vocabulary.load(VOCAB_DIR / "nl_vocab.pkl")
    print(f"  SQL vocab : {len(sql_vocab)}")
    print(f"  NL  vocab : {len(nl_vocab)}")

    # ── BERT tokenizer ────────────────────────────────
    print("\n[2] Loading BERT tokenizer ...")
    from transformers import BertTokenizerFast
    bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    print(f"  Tokenizer : {BERT_MODEL_NAME}")

    # ── Build DataLoaders ─────────────────────────────
    print("\n[3] Building Chinook DataLoaders ...")

    chinook_train = ChinookDataset(
        csv_path      = CHINOOK_CSV,
        sql_vocab     = sql_vocab,
        bert_tokenizer= bert_tokenizer,
        oversample    = OVERSAMPLE_FACTOR,
    )

    # 80/20 split for train/val from chinook_pairs.csv
    total     = len(chinook_train.data)
    val_size  = max(1, int(total * 0.2))
    train_size= total - val_size

    # val uses oversample=1 (no oversampling for validation)
    chinook_val = ChinookDataset(
        csv_path      = CHINOOK_CSV,
        sql_vocab     = sql_vocab,
        bert_tokenizer= bert_tokenizer,
        oversample    = 1,
    )

    # use first train_size rows for train, last val_size for val
    from torch.utils.data import Subset
    train_indices = list(range(train_size * OVERSAMPLE_FACTOR))
    val_indices   = list(range(train_size, total))

    train_ds = Subset(chinook_train, train_indices)
    val_ds   = Subset(chinook_val,   val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size  = FINETUNE_BATCH,
        shuffle     = True,
        collate_fn  = collate_chinook,
        num_workers = 0,
        pin_memory  = torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = FINETUNE_BATCH,
        shuffle     = False,
        collate_fn  = collate_chinook,
        num_workers = 0,
        pin_memory  = torch.cuda.is_available(),
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")

    # ── Build model + load best_bert.pt weights ───────
    print("\n[4] Loading model from best_bert.pt ...")
    model = build_model(
        sql_vocab_size = len(sql_vocab),
        device         = device,
        nl_vocab_size  = len(nl_vocab),
    )

    source_ckpt = torch.load(SOURCE_MODEL, map_location=device)
    model.load_state_dict(source_ckpt["model_state"])
    print(f"  Loaded weights from best_bert.pt "
          f"(epoch {source_ckpt.get('epoch','?')}, "
          f"val_loss {source_ckpt.get('val_loss', source_ckpt.get('best_val_loss','?'))})")
    print(f"  best_bert.pt is NOT modified — fine-tuned model "
          f"will be saved to finetuned_bert.pt")

    # ── Optimizer — very small LR ─────────────────────
    # LR=1e-5 preserves existing SQL syntax knowledge
    # while allowing Chinook-specific patterns to be learned
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = FINETUNE_LR,
        weight_decay = 1e-2,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── Resume fine-tune if checkpoint exists ─────────
    start_epoch   = 1
    best_val_loss = float("inf")
    no_improve    = 0
    tf_ratio      = TEACHER_FORCING
    log_rows      = []

    if RESUME_MODEL.exists():
        print(f"\n  Resuming fine-tune from {RESUME_MODEL} ...")
        resume_ckpt = torch.load(RESUME_MODEL, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch   = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]
        no_improve    = resume_ckpt["no_improve"]
        tf_ratio      = resume_ckpt.get("tf_ratio", TEACHER_FORCING)
        log_rows      = resume_ckpt.get("log_rows", [])
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"\n  Starting fresh fine-tuning ...")

    # ── Fine-tuning loop ──────────────────────────────
    print(f"\n[5] Fine-tuning epochs {start_epoch}→{FINETUNE_EPOCHS} ...")
    print(f"  LR (fine-tune): {FINETUNE_LR}  "
          f"(original was ~1e-4, 10x smaller to preserve knowledge)")
    print(f"  Batch size    : {FINETUNE_BATCH}")
    print(f"  Oversample    : {OVERSAMPLE_FACTOR}x "
          f"(481 pairs → {481 * OVERSAMPLE_FACTOR} per epoch)")
    print(f"  Early stopping: patience={FINETUNE_PATIENCE}\n")

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion,
            device, train=True, tf_ratio=tf_ratio
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion,
            device, train=False
        )

        elapsed   = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl   = math.exp(min(val_loss,   20))

        print(f"  Ep {epoch:02d}/{FINETUNE_EPOCHS} "
              f"| {elapsed:.0f}s "
              f"| train loss={train_loss:.4f} ppl={train_ppl:.1f} "
              f"acc={train_acc:.1f}% "
              f"| val loss={val_loss:.4f} ppl={val_ppl:.1f} "
              f"acc={val_acc:.1f}% "
              f"| tf={tf_ratio:.2f}")

        # show Chinook sample predictions every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            show_chinook_samples(model, sql_vocab, bert_tokenizer, device)

        log_rows.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 4),
            "val_loss"  : round(val_loss,   4),
            "train_ppl" : round(train_ppl,  2),
            "val_ppl"   : round(val_ppl,    2),
            "train_acc" : round(train_acc,  2),
            "val_acc"   : round(val_acc,    2),
            "tf_ratio"  : round(tf_ratio,   3),
            "lr"        : optimizer.param_groups[0]["lr"],
        })

        # save best fine-tuned model (never overwrites best_bert.pt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"       : val_loss,
                "encoder_type"   : ENCODER_TYPE,
                "sql_vocab_size" : len(sql_vocab),
                "nl_vocab_size"  : len(nl_vocab),
                "finetune"       : True,
                "source_model"   : str(SOURCE_MODEL),
                "chinook_pairs"  : 481,
            }, FINETUNE_MODEL)
            print(f"    [SAVED] finetuned_bert.pt (val={val_loss:.4f})")
        else:
            no_improve += 1
            print(f"    No improvement {no_improve}/{FINETUNE_PATIENCE}")
            if no_improve >= FINETUNE_PATIENCE:
                print(f"\n  Early stop at epoch {epoch}")
                break

        # save resume checkpoint
        torch.save({
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss"  : best_val_loss,
            "no_improve"     : no_improve,
            "tf_ratio"       : tf_ratio,
            "log_rows"       : log_rows,
            "encoder_type"   : ENCODER_TYPE,
        }, RESUME_MODEL)

        scheduler.step(val_loss)
        tf_ratio = max(TF_MIN, tf_ratio - TF_DECAY)

    # ── Save log ──────────────────────────────────────
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)
    print(f"\n  Log saved → {LOG_FILE}")

    # ── Final comparison ──────────────────────────────
    print("\n[6] Final Chinook sample comparison ...")
    print("\n  Loading finetuned_bert.pt for final check ...")
    ft_ckpt = torch.load(FINETUNE_MODEL, map_location=device)
    model.load_state_dict(ft_ckpt["model_state"])
    show_chinook_samples(model, sql_vocab, bert_tokenizer, device)

    print("\n" + "="*60)
    print("  Fine-tuning complete!")
    print(f"  Source model     : {SOURCE_MODEL}  ← UNCHANGED")
    print(f"  Fine-tuned model : {FINETUNE_MODEL}  ← NEW")
    print(f"  Best val loss    : {best_val_loss:.4f}")
    print(f"  Training log     : {LOG_FILE}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Push finetuned_bert.pt to shared drive / Tanvi's machine")
    print("  2. Run: python checks/eval_queries.py")
    print("  3. Compare old vs new scores")


if __name__ == "__main__":
    main()
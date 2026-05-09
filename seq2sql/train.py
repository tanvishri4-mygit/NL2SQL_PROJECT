"""
train.py
========
Training loop for NL2SQL Seq2Seq model.
Supports both BERT and BiLSTM encoder modes.
Uses RTX 4070 GPU automatically.

Run from project root:
    python seq2sql/train.py
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

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    ENCODER_TYPE, MODEL_DIR, VOCAB_DIR,
    BATCH_SIZE, N_EPOCHS, CLIP_GRAD,
    TEACHER_FORCING, TF_DECAY, TF_MIN, PATIENCE,
    LR_BERT, LR_BILSTM, PAD_IDX
)
from seq2sql.vocabulary import Vocabulary
from seq2sql.dataset    import get_dataloaders
from seq2sql.model      import build_model

BEST_MODEL    = MODEL_DIR / f"best_{ENCODER_TYPE}.pt"
RESUME_MODEL  = MODEL_DIR / f"resume_{ENCODER_TYPE}.pt"
LOG_FILE      = MODEL_DIR / f"log_{ENCODER_TYPE}.csv"

# ═══════════════════════════════════════════════════════
# Train / eval one epoch
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
        src = src.to(device)
        tgt = tgt.to(device)
        if att_mask is not None:
            att_mask = att_mask.to(device)

        if train:
            optimizer.zero_grad()
            output = model(
                src, att_mask, tgt,
                teacher_forcing_ratio=tf_ratio
            )
        else:
            with torch.no_grad():
                output = model(
                    src, att_mask, tgt,
                    teacher_forcing_ratio=0.0
                )

        B, T, V = output.shape
        loss = criterion(
            output.reshape(B * T, V),
            tgt[:, 1:].reshape(B * T)
        )

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_GRAD
            )
            optimizer.step()

        total_loss += loss.item()

        # token accuracy — ignore PAD tokens
        with torch.no_grad():
            preds   = output.reshape(B * T, V).argmax(dim=1)
            targets = tgt[:, 1:].reshape(B * T)
            mask    = targets != PAD_IDX
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_tokens  += mask.sum().item()

        # live loss in progress bar
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.3f}"})

    avg_loss  = total_loss / len(loader)
    token_acc = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    return avg_loss, token_acc


@torch.no_grad()
def show_samples(model, val_loader, sql_vocab, device, n=3):
    """
    Print n sample question → predicted SQL from validation set.
    Called every 5 epochs so you can see if model is improving.
    """
    model.eval()
    from config import SOS_IDX, EOS_IDX

    # get one batch
    src, att_mask, tgt = next(iter(val_loader))
    src = src[:n].to(device)
    tgt = tgt[:n].to(device)
    if att_mask is not None:
        att_mask = att_mask[:n].to(device)

    print(f"\n  {'─'*51}")
    print(f"  Sample predictions (val set):")
    print(f"  {'─'*51}")

    for i in range(n):
        # greedy generate
        token_ids, _ = model.generate(
            src[i:i+1],
            att_mask[i:i+1] if att_mask is not None else None,
            max_len=60
        )

        # decode predicted SQL
        pred_tokens = sql_vocab.decode(token_ids)
        if EOS_IDX in token_ids:
            pred_tokens = pred_tokens[:token_ids.index(EOS_IDX)]
        pred_tokens = [t for t in pred_tokens
                       if t not in ("<PAD>","<UNK>","<SOS>","<EOS>")]
        pred_sql = " ".join(pred_tokens)

        # decode gold SQL
        gold_ids    = tgt[i].tolist()
        gold_tokens = sql_vocab.decode(gold_ids)
        gold_tokens = [t for t in gold_tokens
                       if t not in ("<PAD>","<UNK>","<SOS>","<EOS>")]
        gold_sql = " ".join(gold_tokens)

        print(f"  [{i+1}] GOLD : {gold_sql[:80]}")
        print(f"       PRED : {pred_sql[:80]}")
        print()

    print(f"  {'─'*51}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print(f"  NL2SQL Training  [{ENCODER_TYPE.upper()} encoder]")
    print("="*55)

    # ── Device ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : "
              f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # ── Load vocabularies ─────────────────────────────
    print("\n[1] Loading vocabularies ...")
    sql_vocab = Vocabulary.load(VOCAB_DIR / "sql_vocab.pkl")
    nl_vocab  = Vocabulary.load(VOCAB_DIR / "nl_vocab.pkl")
    print(f"  SQL vocab : {len(sql_vocab)}")
    print(f"  NL  vocab : {len(nl_vocab)}")

    # ── BERT tokenizer (if needed) ────────────────────
    bert_tokenizer = None
    if ENCODER_TYPE == "bert":
        from transformers import BertTokenizerFast
        from config import BERT_MODEL_NAME
        print(f"\n[2] Loading BERT tokenizer ...")
        bert_tokenizer = BertTokenizerFast.from_pretrained(
            BERT_MODEL_NAME
        )

    # ── DataLoaders ───────────────────────────────────
    print("\n[3] Building DataLoaders ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        sql_vocab      = sql_vocab,
        nl_vocab       = nl_vocab,
        bert_tokenizer = bert_tokenizer,
        batch_size     = BATCH_SIZE,
    )

    # ── Model ─────────────────────────────────────────
    print("\n[4] Building model ...")
    model = build_model(
        sql_vocab_size = len(sql_vocab),
        device         = device,
        nl_vocab_size  = len(nl_vocab),
    )

    # ── Optimizer + loss ──────────────────────────────
    lr = LR_BERT if ENCODER_TYPE == "bert" else LR_BILSTM
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = 1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── Resume from checkpoint if exists ─────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    no_improve    = 0
    tf_ratio      = TEACHER_FORCING
    log_rows      = []

    if RESUME_MODEL.exists():
        print(f"\n  ⚡ Resuming from {RESUME_MODEL} ...")
        resume_ckpt = torch.load(RESUME_MODEL, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch   = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]
        no_improve    = resume_ckpt["no_improve"]
        tf_ratio      = resume_ckpt["tf_ratio"]
        log_rows      = resume_ckpt.get("log_rows", [])
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"\n  Starting fresh training ...")

    # ── Training loop ─────────────────────────────────
    print(f"\n[5] Training epochs {start_epoch}→{N_EPOCHS} ...")
    print(f"  LR             : {lr}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Teacher forcing: {TEACHER_FORCING} → {TF_MIN}")
    print(f"  Early stopping : patience={PATIENCE}\n")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion,
            device, train=True, tf_ratio=tf_ratio
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion,
            device, train=False
        )

        elapsed = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl   = math.exp(min(val_loss,   20))

        print(f"  Ep {epoch:02d}/{N_EPOCHS} "
              f"| {elapsed:.0f}s "
              f"| train loss={train_loss:.4f} ppl={train_ppl:.1f} acc={train_acc:.1f}% "
              f"| val loss={val_loss:.4f} ppl={val_ppl:.1f} acc={val_acc:.1f}% "
              f"| tf={tf_ratio:.2f}")

        # show sample predictions every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            show_samples(model, val_loader, sql_vocab, device, n=3)

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
            }, BEST_MODEL)
            print(f"    ✅ saved (val={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch}")
                break

        # save resume checkpoint every epoch
        # so we can always continue from last completed epoch
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

    # ── Test set ──────────────────────────────────────
    print("\n[6] Test evaluation ...")
    ckpt = torch.load(BEST_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = run_epoch(
        model, test_loader, None, criterion,
        device, train=False
    )
    print(f"  Test loss : {test_loss:.4f}  "
          f"PPL: {math.exp(min(test_loss,20)):.2f}  "
          f"Token Acc: {test_acc:.1f}%")

    print("\n" + "="*55)
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {BEST_MODEL}")
    print("="*55)
    print("\nNext: python seq2sql/inference.py")

if __name__ == "__main__":
    main()

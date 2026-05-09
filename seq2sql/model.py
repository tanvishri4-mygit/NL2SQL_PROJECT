"""
model.py
========
Assembles the full Seq2Seq model.
Picks encoder based on config.ENCODER_TYPE.
Decoder is always the same.

Usage:
    from seq2sql.model import build_model
    model = build_model(sql_vocab_size, nl_vocab_size, device)
"""

import torch
import torch.nn as nn
import random
import sys
from pathlib import Path

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    ENCODER_TYPE, SOS_IDX, EOS_IDX,
    DECODER_HIDDEN_DIM, DECODER_N_LAYERS
)
from seq2sql.decoder import Decoder

# ═══════════════════════════════════════════════════════
# Seq2Seq wrapper
# ═══════════════════════════════════════════════════════

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, attention_mask, tgt,
                teacher_forcing_ratio: float = 0.5):
        """
        src            : (B, src_len)
        attention_mask : (B, src_len) or None  [BERT only]
        tgt            : (B, tgt_len)  includes SOS at pos 0

        Returns:
            outputs : (B, tgt_len-1, vocab_size)
        """
        B          = src.size(0)
        tgt_len    = tgt.size(1)
        vocab_size = self.decoder.sql_vocab_size

        outputs = torch.zeros(B, tgt_len - 1, vocab_size,
                              device=self.device)

        enc_outputs, hidden, cell = self.encoder(
            src, attention_mask
        )

        # reshape hidden/cell for decoder
        hidden = hidden.unsqueeze(0)   # (1,B,dec_dim)
        cell   = cell.unsqueeze(0)

        dec_input = tgt[:, 0]          # SOS tokens : (B,)

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(
                dec_input, enc_outputs, hidden, cell
            )
            outputs[:, t - 1, :] = logits

            use_teacher = random.random() < teacher_forcing_ratio
            top1        = logits.argmax(dim=1)
            dec_input   = tgt[:, t] if use_teacher else top1

        return outputs

    @torch.no_grad()
    def generate(self, src, attention_mask, max_len: int = 100):
        """
        Greedy decoding for inference.
        src            : (1, src_len)
        attention_mask : (1, src_len) or None

        Returns:
            token_ids    : list of predicted SQL token indices
            attn_weights : list of attention weight tensors
        """
        self.eval()
        enc_outputs, hidden, cell = self.encoder(src, attention_mask)
        hidden = hidden.unsqueeze(0)
        cell   = cell.unsqueeze(0)

        dec_input    = torch.tensor([SOS_IDX], device=self.device)
        token_ids    = []
        attn_weights = []

        for _ in range(max_len):
            logits, hidden, cell, attn = self.decoder(
                dec_input, enc_outputs, hidden, cell
            )
            top1 = logits.argmax(dim=1)
            token_ids.append(top1.item())
            attn_weights.append(attn.squeeze(0).cpu())

            if top1.item() == EOS_IDX:
                break

            dec_input = top1

        return token_ids, attn_weights

    @torch.no_grad()
    def generate_beam(self, src, attention_mask,
                      max_len: int = 100, beam_width: int = 5):
        """
        Beam search decoding — keeps top beam_width candidates at each step.

        Why better than greedy:
            Greedy always picks the single highest-probability token.
            If that token leads to a bad sequence (e.g. "CA" instead of "USA"),
            it cannot recover. Beam search explores beam_width paths in parallel
            and returns the one with the highest total log-probability.

        Returns:
            token_ids    : list of token indices for best sequence
            attn_weights : attention weights for best sequence (first beam)
        """
        import math
        self.eval()

        # encode once — shared across all beams
        enc_outputs, hidden, cell = self.encoder(src, attention_mask)
        hidden = hidden.unsqueeze(0)   # (1, 1, dec_dim)
        cell   = cell.unsqueeze(0)

        # each beam is: (log_prob, token_ids, hidden, cell, attn_weights)
        beams = [(0.0, [SOS_IDX], hidden, cell, [])]
        completed = []

        for step in range(max_len):
            if not beams:
                break

            all_candidates = []

            for log_prob, ids, h, c, attns in beams:
                # if this beam already ended, carry it forward
                if ids[-1] == EOS_IDX:
                    completed.append((log_prob, ids, attns))
                    continue

                dec_input = torch.tensor(
                    [ids[-1]], device=self.device
                )
                logits, new_h, new_c, attn = self.decoder(
                    dec_input, enc_outputs, h, c
                )

                # log softmax for numerical stability
                import torch.nn.functional as F
                log_probs = F.log_softmax(logits, dim=1)  # (1, vocab)

                # take top beam_width tokens
                topk_log_probs, topk_ids = log_probs.topk(
                    beam_width, dim=1
                )

                for k in range(beam_width):
                    tok_id   = topk_ids[0, k].item()
                    tok_lp   = topk_log_probs[0, k].item()
                    new_lp   = log_prob + tok_lp
                    new_ids  = ids + [tok_id]
                    new_attn = attns + [attn.squeeze(0).cpu()]

                    all_candidates.append(
                        (new_lp, new_ids, new_h, new_c, new_attn)
                    )

            # length-normalise scores to avoid bias toward short sequences
            # score = log_prob / length ^ length_penalty
            length_penalty = 0.7

            def norm_score(cand):
                lp, ids, _, _, _ = cand
                length = max(len(ids) - 1, 1)  # exclude SOS
                return lp / (length ** length_penalty)

            # keep only top beam_width candidates
            all_candidates.sort(key=norm_score, reverse=True)
            beams = all_candidates[:beam_width]

            # move any completed beams out
            still_running = []
            for cand in beams:
                lp, ids, h, c, attns = cand
                if ids[-1] == EOS_IDX:
                    completed.append((lp, ids, attns))
                else:
                    still_running.append(cand)
            beams = still_running

        # add any remaining running beams to completed
        for lp, ids, h, c, attns in beams:
            completed.append((lp, ids, attns))

        if not completed:
            # fallback to greedy if something went wrong
            return self.generate(src, attention_mask, max_len)

        # pick best completed sequence by length-normalised score
        def final_score(cand):
            lp, ids, _ = cand
            length = max(len(ids) - 1, 1)
            return lp / (length ** length_penalty)

        best_lp, best_ids, best_attns = max(completed, key=final_score)

        # strip SOS and EOS
        result_ids = [t for t in best_ids
                      if t not in (SOS_IDX, EOS_IDX)]

        return result_ids, best_attns


# ═══════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════

def build_model(
    sql_vocab_size : int,
    device         : torch.device,
    nl_vocab_size  : int = None,   # only needed for bilstm
) -> Seq2Seq:
    """
    Builds and returns the full Seq2Seq model on device.
    Encoder chosen based on config.ENCODER_TYPE.
    """
    print(f"\n  Building model (encoder={ENCODER_TYPE}) ...")

    if ENCODER_TYPE == "bert":
        from seq2sql.encoder_bert import BERTEncoder
        encoder = BERTEncoder()
        enc_dim = encoder.output_dim   # 768

    else:
        from seq2sql.encoder_bilstm import BiLSTMEncoder
        if nl_vocab_size is None:
            raise ValueError(
                "nl_vocab_size required for bilstm encoder"
            )
        encoder = BiLSTMEncoder(vocab_size=nl_vocab_size)
        enc_dim = encoder.output_dim   # hidden*2

    decoder = Decoder(
        sql_vocab_size = sql_vocab_size,
        enc_dim        = enc_dim
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    # Xavier init for decoder weights
    for name, param in decoder.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    total = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
    print(f"  Total trainable params: {total:,}")

    return model
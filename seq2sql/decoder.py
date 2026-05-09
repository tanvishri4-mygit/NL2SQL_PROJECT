"""
decoder.py
==========
LSTM decoder with Bahdanau attention.
Identical for both BERT and BiLSTM encoders.

At each step:
    1. Embed previous token
    2. Compute attention over encoder outputs
    3. LSTM([embedded; context]) → output
    4. Project to SQL vocab distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    DECODER_EMBED_DIM, DECODER_HIDDEN_DIM,
    DECODER_N_LAYERS, DECODER_DROPOUT, PAD_IDX
)

# ═══════════════════════════════════════════════════════
# Bahdanau Attention
# ═══════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """
    Additive attention.
    score(h_t, h_s) = v · tanh(W1·h_t + W2·h_s)
    """
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, dec_dim, bias=False)
        self.W2 = nn.Linear(enc_dim, dec_dim, bias=False)
        self.v  = nn.Linear(dec_dim, 1,       bias=False)

    def forward(self, dec_hidden, enc_outputs):
        """
        dec_hidden  : (B, dec_dim)
        enc_outputs : (B, src_len, enc_dim)
        Returns:
            context      : (B, enc_dim)
            attn_weights : (B, src_len)
        """
        src_len = enc_outputs.size(1)
        # expand dec_hidden over src_len
        h = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
        # h : (B, src_len, dec_dim)

        energy = torch.tanh(self.W1(h) + self.W2(enc_outputs))
        # energy : (B, src_len, dec_dim)

        scores = self.v(energy).squeeze(-1)
        # scores : (B, src_len)

        attn_weights = F.softmax(scores, dim=1)
        # attn_weights : (B, src_len)

        context = torch.bmm(
            attn_weights.unsqueeze(1), enc_outputs
        ).squeeze(1)
        # context : (B, enc_dim)

        return context, attn_weights


# ═══════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self, sql_vocab_size: int, enc_dim: int):
        """
        sql_vocab_size : size of SQL output vocabulary
        enc_dim        : encoder output dimension
                         (768 for BERT, hidden*2 for BiLSTM)
        """
        super().__init__()

        self.sql_vocab_size = sql_vocab_size
        self.attention      = BahdanauAttention(enc_dim, DECODER_HIDDEN_DIM)

        self.embedding = nn.Embedding(
            sql_vocab_size, DECODER_EMBED_DIM,
            padding_idx=PAD_IDX
        )
        # input to LSTM = embedded token + context vector
        self.rnn = nn.LSTM(
            DECODER_EMBED_DIM + enc_dim,
            DECODER_HIDDEN_DIM,
            num_layers  = DECODER_N_LAYERS,
            dropout     = DECODER_DROPOUT if DECODER_N_LAYERS > 1 else 0.0,
            batch_first = True,
        )
        # output projection: dec_out + context + embedded → vocab
        self.fc_out = nn.Linear(
            DECODER_HIDDEN_DIM + enc_dim + DECODER_EMBED_DIM,
            sql_vocab_size
        )
        self.dropout = nn.Dropout(DECODER_DROPOUT)

        total = sum(p.numel() for p in self.parameters()
                    if p.requires_grad)
        print(f"  Decoder params: {total:,}")

    def forward(self, token, enc_outputs, hidden, cell):
        """
        token       : (B,)
        enc_outputs : (B, src_len, enc_dim)
        hidden      : (n_layers, B, dec_dim)
        cell        : (n_layers, B, dec_dim)

        Returns:
            logits       : (B, vocab_size)
            hidden       : (n_layers, B, dec_dim)
            cell         : (n_layers, B, dec_dim)
            attn_weights : (B, src_len)
        """
        token    = token.unsqueeze(1)                    # (B,1)
        embedded = self.dropout(self.embedding(token))   # (B,1,emb)

        context, attn_weights = self.attention(
            hidden[-1], enc_outputs
        )
        context_expanded = context.unsqueeze(1)          # (B,1,enc_dim)

        rnn_input = torch.cat([embedded, context_expanded], dim=2)
        # (B,1, emb+enc_dim)

        rnn_out, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # rnn_out : (B,1,dec_dim)

        rnn_out  = rnn_out.squeeze(1)   # (B, dec_dim)
        embedded = embedded.squeeze(1)  # (B, emb)

        logits = self.fc_out(
            torch.cat([rnn_out, context, embedded], dim=1)
        )
        # logits : (B, vocab_size)

        return logits, hidden, cell, attn_weights

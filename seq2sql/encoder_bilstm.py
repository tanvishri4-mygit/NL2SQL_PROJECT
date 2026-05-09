"""
encoder_bilstm.py
=================
Bidirectional LSTM encoder — fallback if professor says no to BERT.
Trained completely from scratch. No pretrained weights.

Input : token indices (B, src_len)
Output: encoder_outputs (B, src_len, hidden*2)
        hidden          (B, hidden)
        cell            (B, hidden)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    BILSTM_EMBED_DIM, BILSTM_HIDDEN_DIM,
    BILSTM_N_LAYERS, BILSTM_DROPOUT,
    DECODER_HIDDEN_DIM, PAD_IDX
)

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, BILSTM_EMBED_DIM,
            padding_idx=PAD_IDX
        )
        self.rnn = nn.LSTM(
            BILSTM_EMBED_DIM,
            BILSTM_HIDDEN_DIM,
            num_layers    = BILSTM_N_LAYERS,
            bidirectional = True,
            dropout       = BILSTM_DROPOUT if BILSTM_N_LAYERS > 1 else 0.0,
            batch_first   = True,
        )
        # project bidirectional (hidden*2) → decoder hidden dim
        self.fc_h = nn.Linear(BILSTM_HIDDEN_DIM * 2, DECODER_HIDDEN_DIM)
        self.fc_c = nn.Linear(BILSTM_HIDDEN_DIM * 2, DECODER_HIDDEN_DIM)
        self.dropout = nn.Dropout(BILSTM_DROPOUT)

        total = sum(p.numel() for p in self.parameters()
                    if p.requires_grad)
        print(f"  BiLSTM encoder params: {total:,}")

    def forward(self, src, attention_mask=None):
        """
        src  : (B, src_len)
        attention_mask : ignored (exists for interface compatibility with BERT)

        Returns:
            encoder_outputs : (B, src_len, hidden*2)
            hidden          : (B, DECODER_HIDDEN_DIM)
            cell            : (B, DECODER_HIDDEN_DIM)
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # concat last layer fwd + bwd
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        c = torch.cat([cell[-2],   cell[-1]],   dim=1)

        hidden = torch.tanh(self.fc_h(h))
        cell   = torch.tanh(self.fc_c(c))

        return outputs, hidden, cell

    @property
    def output_dim(self):
        return BILSTM_HIDDEN_DIM * 2

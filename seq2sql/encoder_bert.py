"""
encoder_bert.py
===============
BERT-based encoder for NL2SQL.

Input : input_ids (B, seq_len) + attention_mask (B, seq_len)
        contains: [CLS] question [SEP] schema [SEP]

Output: encoder_outputs (B, seq_len, 768)
        hidden          (B, 512)   — projected for decoder
        cell            (B, 512)   — projected for decoder
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
    BERT_MODEL_NAME, BERT_HIDDEN_DIM,
    BERT_FREEZE_LAYERS, DECODER_HIDDEN_DIM
)

class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        from transformers import BertModel
        print(f"  Loading BERT: {BERT_MODEL_NAME} ...")
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

        # freeze bottom N layers to save GPU memory + training time
        # only fine-tune top layers
        modules_to_freeze = [
            self.bert.embeddings,
            *self.bert.encoder.layer[:BERT_FREEZE_LAYERS]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        frozen  = sum(1 for p in self.bert.parameters()
                      if not p.requires_grad)
        trainable = sum(1 for p in self.bert.parameters()
                        if p.requires_grad)
        print(f"  BERT frozen params : {frozen}")
        print(f"  BERT trainable     : {trainable}")

        # project BERT hidden (768) → decoder hidden (512)
        self.fc_h = nn.Linear(BERT_HIDDEN_DIM, DECODER_HIDDEN_DIM)
        self.fc_c = nn.Linear(BERT_HIDDEN_DIM, DECODER_HIDDEN_DIM)

    def forward(self, input_ids, attention_mask):
        """
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)

        Returns:
            encoder_outputs : (B, seq_len, 768)
            hidden          : (B, 512)
            cell            : (B, 512)
        """
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )

        # all token hidden states
        encoder_outputs = outputs.last_hidden_state
        # (B, seq_len, 768)

        # [CLS] token → summary of entire input → use as initial
        # decoder hidden/cell state
        cls_output = outputs.pooler_output
        # (B, 768)

        hidden = torch.tanh(self.fc_h(cls_output))  # (B, 512)
        cell   = torch.tanh(self.fc_c(cls_output))  # (B, 512)

        return encoder_outputs, hidden, cell

    @property
    def output_dim(self):
        """Dimension of encoder output per token."""
        return BERT_HIDDEN_DIM

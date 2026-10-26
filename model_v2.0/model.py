"""
Model v2.0 — Model Architecture
DistilBERT + sigmoid head for multi-label CAPEC classification.

Key difference from v1:
- v1: softmax over 2 classes (binary benign/malicious)
- v2: sigmoid per label (13 independent binary outputs)
- Loss: BCEWithLogitsLoss (not CrossEntropyLoss)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from config import BASE_MODEL, NUM_LABELS, DROPOUT


class MultiLabelWAFModel(nn.Module):
    """DistilBERT + multi-label classification head.

    Architecture:
        DistilBERT (6 layers, 768 dim)
        → [CLS] token representation (768)
        → Dropout (0.3)
        → Linear (768 → 13)
        → Sigmoid (applied in loss function via BCEWithLogitsLoss)

    Outputs raw logits — apply sigmoid yourself at inference time.
    """

    def __init__(self, model_name: str = BASE_MODEL, num_labels: int = NUM_LABELS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, num_labels) float tensor, optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        logits = self.classifier(cls_output)  # (batch, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_weighted_loss(self, logits, labels, pos_weight):
        """BCEWithLogitsLoss with per-class positive weights for imbalanced data."""
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, labels)


def create_model(model_name: str = BASE_MODEL, num_labels: int = NUM_LABELS) -> MultiLabelWAFModel:
    """Factory function to create a fresh model."""
    model = MultiLabelWAFModel(model_name=model_name, num_labels=num_labels)
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model created: {param_count:,} params ({trainable:,} trainable)")
    return model

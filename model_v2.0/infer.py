"""
Model v2.0 — Inference
Standalone inference class — drop-in compatible with 4thWall API.
"""

import json
import os

import torch
import numpy as np
from transformers import AutoTokenizer

from config import BASE_MODEL, MAX_LENGTH, SIGMOID_THRESHOLD, CAPEC_LABELS, CAPEC_DESCRIPTIONS
from model import MultiLabelWAFModel
from preprocess import clean_http_text


class MultiLabelWAFClassifier:
    """Load a trained multi-label model and run predictions.

    Drop-in replacement for v1's WAFClassifier — same .predict() / .predict_batch() API
    but returns multi-label results instead of binary.
    """

    def __init__(self, model_dir: str, threshold: float = SIGMOID_THRESHOLD):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # Load label columns
        label_path = os.path.join(model_dir, "label_columns.json")
        if os.path.exists(label_path):
            with open(label_path) as f:
                self.label_columns = json.load(f)
        else:
            self.label_columns = CAPEC_LABELS

        self.num_labels = len(self.label_columns)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Load model
        self.model = MultiLabelWAFModel(num_labels=self.num_labels)
        state_dict = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load eval report if available
        self.metrics = None
        report_path = os.path.join(model_dir, "eval_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self.metrics = json.load(f)

    def predict(self, http_text: str) -> dict:
        """Predict labels for a single HTTP request.

        Args:
            http_text: raw HTTP request text (or pre-cleaned)

        Returns:
            {
                "scores": {label: float, ...},       # sigmoid probability per label
                "labels": [str, ...],                # labels above threshold
                "is_malicious": bool,                # True if any attack label triggered
                "top_attack": str | None,            # highest-scoring attack label
                "max_score": float,                  # highest attack score
            }
        """
        return self.predict_batch([http_text])[0]

    def predict_batch(self, http_texts: list) -> list:
        """Batch prediction."""
        if not http_texts:
            return []

        # Clean texts
        cleaned = [clean_http_text(t) for t in http_texts]

        # Tokenize
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()

        # Build results
        results = []
        for i in range(len(http_texts)):
            scores = {self.label_columns[j]: float(probs[i, j])
                      for j in range(self.num_labels)}

            triggered = [self.label_columns[j]
                         for j in range(self.num_labels)
                         if probs[i, j] >= self.threshold]

            # Attack labels = everything except "Normal"
            attack_scores = {k: v for k, v in scores.items()
                             if k.lower() != "normal"}
            top_attack = max(attack_scores, key=attack_scores.get) if attack_scores else None
            max_attack_score = max(attack_scores.values()) if attack_scores else 0.0

            is_malicious = any(
                k.lower() != "normal" and v >= self.threshold
                for k, v in scores.items()
            )

            results.append({
                "scores": scores,
                "labels": triggered,
                "is_malicious": is_malicious,
                "top_attack": top_attack,
                "max_score": max_attack_score,
            })

        return results

    def predict_v1_compatible(self, request_text: str) -> dict:
        """v1-compatible output format for drop-in replacement.

        Returns:
            {
                "score": float (max attack probability),
                "label": "benign" or "malicious",
                "confidence": float,
                "attack_types": [str, ...],  # NEW: which CAPEC categories
            }
        """
        result = self.predict(request_text)
        score = result["max_score"]
        label = "malicious" if result["is_malicious"] else "benign"
        confidence = score if label == "malicious" else (1.0 - score)
        attack_types = [l for l in result["labels"] if l.lower() != "normal"]

        return {
            "score": score,
            "label": label,
            "confidence": confidence,
            "attack_types": attack_types,
        }

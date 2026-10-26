"""
WAFClassifier: High-performance inference engine for 4thwall_v3 WAF.
Supports Frozen Model v2.1b PyTorch checkpoints and HuggingFace Hub repositories.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from ml.config import (
    MODEL_NAME,
    HF_MODEL_REPO,
    LOCAL_MODEL_PATH,
    FROZEN_MODEL_PATH,
    MAX_LENGTH,
    MAX_BODY_CHARS,
    LABEL_COLS,
    BLOCK_CLASSES,
    ALERT_CLASSES,
    CALIBRATED_THRESHOLDS,
)
from ml.preprocess import serialize_request

logger = logging.getLogger(__name__)


class PyTorchFrozenBackbone(nn.Module):
    """Fallback wrapper for raw v2_1b_best.pt state dict."""
    def __init__(self, base_model: str = MODEL_NAME, num_labels: int = 13, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        # Mimic HuggingFace sequence classification output
        return type("Output", (), {"logits": logits})()


class WAFClassifier:
    """
    Enterprise-grade WAF Classifier supporting:
    - HuggingFace Hub repo IDs (e.g. 'viitheone/4thwall-waf-v2.1b')
    - Local HuggingFace directories (containing config.json, model.safetensors)
    - Local PyTorch frozen checkpoints ('v2_1b_best.pt')
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if not model_path:
            model_path = os.getenv("MODEL_PATH", HF_MODEL_REPO)

        self.model_path = model_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Optimize CPU multi-threading
        if self.device.type == "cpu":
            num_threads = int(os.getenv("TORCH_THREADS", "4"))
            torch.set_num_threads(num_threads)

        self.thresholds = dict(CALIBRATED_THRESHOLDS)
        self.metrics = self._load_default_metrics()

        # Load model and tokenizer
        self._load_model_and_tokenizer(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "WAFClassifier successfully initialized on %s (source: %s, classes: %d)",
            self.device,
            self.model_path,
            len(LABEL_COLS)
        )

    def _load_model_and_tokenizer(self, path: str):
        """Attempt to load from HuggingFace Hub, local HF directory, or raw PT checkpoint."""
        # 1. Check if path is a raw PyTorch checkpoint or directory containing v2_1b_best.pt
        is_pt_file = os.path.isfile(path) and path.endswith(".pt")
        pt_dir_file = os.path.join(path, "v2_1b_best.pt") if os.path.isdir(path) else None
        
        if is_pt_file or (pt_dir_file and os.path.isfile(pt_dir_file)):
            ckpt_path = path if is_pt_file else pt_dir_file
            ckpt_dir = os.path.dirname(ckpt_path)
            logger.info("Loading PyTorch frozen checkpoint from %s", ckpt_path)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = PyTorchFrozenBackbone(MODEL_NAME, num_labels=len(LABEL_COLS))
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self._load_thresholds_from_dir(ckpt_dir)
            self._load_metadata_from_dir(ckpt_dir)
            return

        # 2. Try loading via HuggingFace AutoModel with remote code support
        try:
            logger.info("Loading HuggingFace model from %s", path)
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)
            
            # Extract thresholds from model config if present
            cfg_thresh = getattr(self.model.config, "thresholds", None)
            if cfg_thresh and isinstance(cfg_thresh, dict):
                self.thresholds.update(cfg_thresh)
            elif os.path.isdir(path):
                self._load_thresholds_from_dir(path)
                self._load_metadata_from_dir(path)
            return
        except Exception as hf_err:
            logger.warning("Primary load from %s failed: %s. Attempting fallback...", path, hf_err)

        # 3. Fallback to local HuggingFace export directory
        if os.path.isdir(LOCAL_MODEL_PATH):
            try:
                logger.info("Fallback: Loading local HF directory %s", LOCAL_MODEL_PATH)
                self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
                self._load_thresholds_from_dir(LOCAL_MODEL_PATH)
                self._load_metadata_from_dir(LOCAL_MODEL_PATH)
                return
            except Exception as e:
                logger.warning("Fallback to %s failed: %s", LOCAL_MODEL_PATH, e)

        # 4. Fallback to local frozen checkpoint directory
        frozen_file = os.path.join(FROZEN_MODEL_PATH, "v2_1b_best.pt")
        if os.path.isfile(frozen_file):
            logger.info("Fallback: Loading local frozen checkpoint %s", frozen_file)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = PyTorchFrozenBackbone(MODEL_NAME, num_labels=len(LABEL_COLS))
            state_dict = torch.load(frozen_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self._load_thresholds_from_dir(FROZEN_MODEL_PATH)
            self._load_metadata_from_dir(FROZEN_MODEL_PATH)
            return

        raise RuntimeError(f"Unable to load WAF model from '{path}' or any local fallback directory.")

    def _load_thresholds_from_dir(self, dir_path: str):
        t_file = os.path.join(dir_path, "thresholds.json")
        if os.path.isfile(t_file):
            try:
                with open(t_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.thresholds.update(data)
                logger.info("Loaded calibrated thresholds from %s", t_file)
            except Exception as e:
                logger.warning("Failed to parse thresholds from %s: %s", t_file, e)

    def _load_metadata_from_dir(self, dir_path: str):
        m_file = os.path.join(dir_path, "metadata.json")
        if os.path.isfile(m_file):
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics.update(data)
                logger.info("Loaded metadata and metrics from %s", m_file)
            except Exception as e:
                logger.warning("Failed to parse metadata from %s: %s", m_file, e)

    @staticmethod
    def _load_default_metrics() -> Dict[str, Any]:
        """Validated benchmarks from Frozen Model v2.1b evaluation."""
        return {
            "model_version": "v2.1b",
            "architecture": "distilbert-base-uncased",
            "classes": 13,
            "macro_f1": 0.9830,
            "macro_f1_excl_272": 0.9923,
            "micro_f1": 0.9912,
            "exploit_catch_rate": 0.9941,
            "benign_fpr": 0.00782,
            "benign_block_rate": 0.00737,
            "accuracy": 0.9912,
            "precision": 0.9923,
            "recall": 0.9941,
            "f1": 0.9830,
        }

    def predict(self, request_text: str) -> Dict[str, Any]:
        """Score a single serialized request string."""
        results = self.predict_batch([request_text])
        return results[0]

    def predict_batch(self, request_texts: List[str]) -> List[Dict[str, Any]]:
        """Batch multi-label prediction for high-throughput proxy inspection."""
        if not request_texts:
            return []

        enc = self.tokenizer(
            request_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for i, text in enumerate(request_texts):
            prob_row = probs[i]
            prob_dict = {LABEL_COLS[j]: float(prob_row[j]) for j in range(len(LABEL_COLS))}

            # Determine firing labels exceeding per-class calibrated threshold
            firing_labels = [
                LABEL_COLS[j] for j in range(len(LABEL_COLS))
                if prob_row[j] >= self.thresholds.get(LABEL_COLS[j], 0.50)
            ]

            # Split into BLOCK vs ALERT tiers
            blocking_labels = [l for l in firing_labels if l in BLOCK_CLASSES]
            alert_labels = [l for l in firing_labels if l in ALERT_CLASSES]

            if blocking_labels:
                decision = "BLOCK"
            elif alert_labels:
                decision = "ALERT"
            else:
                decision = "ALLOW"

            # Determine whether any attack vector is present
            is_attack = bool(blocking_labels or alert_labels or any(l != "000 - Normal" for l in firing_labels))

            # Highest scoring attack vector
            attack_probs = {k: v for k, v in prob_dict.items() if k != "000 - Normal"}
            primary_attack = max(attack_probs, key=attack_probs.get) if attack_probs else None
            max_attack_score = max(attack_probs.values()) if attack_probs else 0.0

            # Backward-compatible fields
            label = "malicious" if is_attack else "benign"
            confidence = max_attack_score if is_attack else prob_dict.get("000 - Normal", 1.0 - max_attack_score)

            results.append({
                "score": float(max_attack_score),
                "label": label,
                "confidence": float(confidence),
                "decision": decision,
                "is_attack": is_attack,
                "firing_labels": firing_labels,
                "blocking_labels": blocking_labels,
                "alert_labels": alert_labels,
                "primary_attack": primary_attack if is_attack else None,
                "probabilities": prob_dict,
                "serialized_text": text,
            })

        return results

    def score_request(
        self,
        method: str = "GET",
        path: str = "/",
        ua: str = "",
        ct: str = "",
        body: str = "",
        max_body_chars: int = MAX_BODY_CHARS
    ) -> Dict[str, Any]:
        """Convenience method to serialize HTTP components and score."""
        text = serialize_request(method=method, path=path, ua=ua, ct=ct, body=body, max_body_chars=max_body_chars)
        return self.predict(text)

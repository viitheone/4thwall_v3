import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.config import MAX_LENGTH


class WAFClassifier:
    def __init__(self, model_path: str):
        """Load model from a HuggingFace Hub repo ID or a local directory path.

        ``model_path`` is passed directly to ``from_pretrained`` which
        transparently handles both Hub repo IDs (e.g. "viitheone/waf-model")
        and local filesystem paths.
        """

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.metrics = self._load_metrics(model_path)

    @staticmethod
    def _load_metrics(model_path: str) -> dict | None:
        """Try to load metrics.json from local path or HuggingFace Hub."""
        import os, json

        # 1. Try local file
        local_metrics = os.path.join(model_path, "metrics.json")
        if os.path.isfile(local_metrics):
            try:
                with open(local_metrics, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        # 2. Try downloading from HuggingFace Hub (model_path is a repo ID)
        try:
            from huggingface_hub import hf_hub_download
            metrics_path = hf_hub_download(
                repo_id=model_path,
                filename="metrics.json",
            )
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

        return None

    def predict(self, request_text: str) -> dict:
        out = self.predict_batch([request_text])
        return out[0]

    def predict_batch(self, request_texts: list) -> list:
        """Batch prediction for efficiency."""
        if not request_texts:
            return []

        enc = self.tokenizer(
            request_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1)

        results = []
        for i in range(len(request_texts)):
            prob_malicious = probs[i, 1].item()
            label = "malicious" if prob_malicious >= 0.5 else "benign"
            results.append({
                "score": prob_malicious,
                "label": label,
                "confidence": prob_malicious if label == "malicious" else (1.0 - prob_malicious),
            })
        return results


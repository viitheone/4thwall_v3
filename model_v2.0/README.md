# Model v2.0 — Multi-Label CAPEC WAF Classifier

A fresh DistilBERT-based multi-label web attack classifier trained on the **SR-BH 2020** real honeypot dataset.

> **Paper**: Riera et al. 2022 — *"A new multi-label dataset for Web attacks CAPEC classification using machine learning techniques"*
> **Dataset**: [Harvard Dataverse (doi:10.7910/DVN/OGOIXX)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OGOIXX)

## What's Different from v1

| | v1 (deprecated) | v2 (this) |
|---|---|---|
| **Training data** | 5,000 synthetic activity logs | 907K real honeypot HTTP requests |
| **Classification** | Binary (benign/malicious) | Multi-label (13 CAPEC categories) |
| **Output layer** | Softmax over 2 classes | Sigmoid per label (independent) |
| **Loss function** | CrossEntropyLoss | BCEWithLogitsLoss (class-weighted) |
| **Base model** | Fine-tuned DistilBERT (overfit) | Fresh `distilbert-base-uncased` |
| **Overfitting checks** | None (single val split) | Early stopping + train/val loss curves |
| **Evaluation** | Accuracy only | Per-label P/R/F1, hamming loss, subset accuracy |

## 13 CAPEC Attack Categories

| Label | Attack Type |
|---|---|
| Normal | Legitimate traffic |
| CAPEC-66 | SQL Injection |
| CAPEC-86 | Cross-Site Scripting (XSS) |
| CAPEC-88 | OS Command Injection |
| CAPEC-126 | Path Traversal |
| CAPEC-228 | Resource Manipulation / DTD Injection |
| CAPEC-231 | Oversized Data / HTTP Flooding |
| CAPEC-234 | Session Hijacking |
| CAPEC-244 | XML External Entity (XXE) |
| CAPEC-257 | Fuzzing for App Mapping |
| CAPEC-541 | Application Fingerprinting |
| CAPEC-550 | URI/URL Scheme Identification |
| CAPEC-664 | Server-Side Request Forgery (SSRF) |

## Setup

```bash
cd model_v2.0
pip install -r requirements.txt
```

## Data

1. Download the SR-BH 2020 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OGOIXX)
2. Place the CSV/TSV file in `data/`

## Train

```bash
python train.py --data_path data/<your_dataset_file>.csv
```

Optional args:
- `--epochs 10` (default 10, with early stopping patience=3)
- `--batch_size 16`
- `--lr 2e-5`
- `--output_dir outputs/`
- `--sep "\t"` (if TSV)

## Evaluate

```bash
python evaluate.py --data_path data/<your_dataset_file>.csv
```

Outputs: per-label P/R/F1, micro/macro averages, hamming loss, subset accuracy, train vs val loss curve analysis.

## Inference

```python
from infer import MultiLabelWAFClassifier

clf = MultiLabelWAFClassifier("outputs/")
result = clf.predict("GET /admin?id=1' OR '1'='1 HTTP/1.1")

print(result["is_malicious"])   # True
print(result["labels"])         # ['CAPEC-66']
print(result["scores"])         # {'Normal': 0.02, 'CAPEC-66': 0.97, ...}

# v1-compatible output
compat = clf.predict_v1_compatible("GET /etc/passwd HTTP/1.1")
print(compat)  # {'score': 0.95, 'label': 'malicious', 'confidence': 0.95, 'attack_types': ['CAPEC-126']}
```

## Architecture

```
Input: "GET /admin?id=1' OR '1'='1 HTTP/1.1\nHost: example.com"
  |
  v
DistilBERT Tokenizer (max 512 tokens)
  |
  v
DistilBERT (6 layers, 768 dim) -- full fine-tune
  |
  v
[CLS] token (768-dim)
  |
  v
Dropout (0.3) -> Linear (768 -> 13) -> Sigmoid
  |
  v
Output: [0.02, 0.97, 0.03, 0.01, ...] (13 independent probabilities)
```

## File Structure

```
model_v2.0/
├── config.py           # Hyperparameters, CAPEC label definitions
├── preprocess.py       # HTTP text cleaning, multi-label encoding
├── dataset.py          # PyTorch Dataset, stratified multi-label split
├── model.py            # DistilBERT + sigmoid head architecture
├── train.py            # Training loop with early stopping
├── evaluate.py         # Per-label metrics, loss curve analysis
├── infer.py            # Standalone inference class
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Dataset storage (not tracked)
└── outputs/            # Trained model + metrics (not tracked)
```

## Citation

```bibtex
@article{riera2022new,
  title={A new multi-label dataset for Web attacks CAPEC classification using machine learning techniques},
  author={Riera, Tania San and others},
  journal={Computers \& Security},
  year={2022},
  publisher={Elsevier}
}
```

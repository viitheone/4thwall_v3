# Quick Setup Guide — Model v2.0

Hey! This folder contains everything needed to train a **web attack classifier**.
Here's how to get it running on your machine.

---

## What This Does

Trains a DistilBERT model to classify HTTP requests into 13 categories:
- **Normal** (legit traffic)
- **12 attack types** (SQL injection, XSS, path traversal, etc.)

## Requirements

- **Python 3.10+**
- **GPU recommended** (CUDA) — training on CPU will take forever with 907K samples
- **~16 GB RAM** minimum
- **~5 GB disk** for model + dataset

## Step-by-Step Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **If you have a CUDA GPU**, install PyTorch with CUDA support first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```
> Check with: `python -c "import torch; print(torch.cuda.is_available())"` — should print `True`.

### 3. Place the dataset

Put the SR-BH 2020 CSV/TSV file in the `data/` folder:

```
model_v2.0/
└── data/
    └── <your_dataset_file>.csv    ← put it here
```

### 4. Train

```bash
python train.py --data_path data/data_capec_multilabel.csv
```

**What you'll see:**
```
  Loading dataset from data/SR-BH-2020.csv...
  Loaded 907815 rows, 15 columns
  Extracting text: 100%|██████████████████████████████| 907815/907815 [02:30<00:00]
  Encoding labels: 100%|██████████████████████████████| 907815/907815 [01:15<00:00]

  Model created: 66,955,789 params (66,955,789 trainable)

  Epoch 1/10 [TRAIN]: 100%|██████████████████████████| 39720/39720 [45:00<00:00, loss=0.0832, avg=0.1204, lr=1.8e-05]
           [VAL]  : 100%|██████████████████████████████| 8506/8506 [05:00<00:00]

  Epoch 1/10 (3000s)
    Train loss: 0.1204
    Val loss:   0.0945
    Micro-F1:   0.8723
    Macro-F1:   0.7456
    -> Saved best model (Macro-F1=0.7456)
```

**Optional flags:**
- `--epochs 10` — max epochs (default 10, stops early if no improvement)
- `--batch_size 16` — decrease to 8 if you run out of GPU memory
- `--lr 2e-5` — learning rate
- `--sep "\t"` — use if your dataset is tab-separated

### 5. Evaluate

After training finishes:

```bash
python evaluate.py --data_path data/data_capec_multilabel.csv
```

This prints per-label precision/recall/F1 and saves results to `outputs/eval_report.json`.

### 6. Send back results

Zip and send back the `outputs/` folder. It should contain:
- `model.pt` — trained model weights (~268 MB)
- `training_history.json` — loss curves per epoch
- `eval_report.json` — test set metrics
- `label_columns.json` — label mapping

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce batch size: `--batch_size 8` or `--batch_size 4` |
| `ModuleNotFoundError` | Make sure you activated the venv and ran `pip install -r requirements.txt` |
| `FileNotFoundError` for dataset | Check the `--data_path` points to the actual file in `data/` |
| Training is super slow | Confirm GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"` |
| `killed` or process crashes | Not enough RAM. Close other apps or use a machine with 16+ GB RAM |

## Estimated Training Time

| Hardware | ~Time per epoch | Total (~10 epochs) |
|---|---|---|
| RTX 3060 / 3070 | ~30-45 min | ~5-7 hrs |
| RTX 3080 / 3090 | ~15-25 min | ~3-4 hrs |
| RTX 4090 | ~10-15 min | ~2-3 hrs |
| CPU only | ~6-8 hrs | ~60-80 hrs (don't) |

Thanks for helping out! 🙏

"""
Model v2.0 — Evaluation
Comprehensive metrics on held-out test set + loss curve analysis.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, classification_report,
    multilabel_confusion_matrix,
)

from config import OUTPUT_DIR, SIGMOID_THRESHOLD, BASE_MODEL, MAX_LENGTH, NUM_LABELS
from dataset import create_dataloaders
from model import MultiLabelWAFModel


def evaluate_model(data_path: str, output_dir: str = OUTPUT_DIR, sep=None, split_mode=None):
    """Run full evaluation on the test split."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Model v2.0 — Evaluation")
    print(f"{'='*60}")

    # Load data (same splits as training due to fixed seed)
    _, _, test_loader, label_columns, _ = create_dataloaders(data_path, sep=sep, split_mode=split_mode)

    # Load label columns from training
    label_path = os.path.join(output_dir, "label_columns.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_columns = json.load(f)

    num_labels = len(label_columns)

    # Load model
    model = MultiLabelWAFModel(num_labels=num_labels)
    state_dict = torch.load(os.path.join(output_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Model loaded from {output_dir}")

    # Run predictions
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            preds = (probs >= SIGMOID_THRESHOLD).astype(int)

            all_preds.append(preds)
            all_labels.append(labels.numpy().astype(int))
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    print(f"\n  Test samples: {len(all_labels)}")

    # ─── Aggregate metrics ─────────────────────────────────
    micro_p = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_r = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    h_loss = hamming_loss(all_labels, all_preds)

    # Subset accuracy (exact match)
    subset_acc = np.mean(np.all(all_preds == all_labels, axis=1))

    print(f"\n  {'='*50}")
    print(f"  EVALUATION RESULTS (Test Set)")
    print(f"  {'='*50}")
    print(f"\n  Micro-Precision:  {micro_p:.4f}")
    print(f"  Micro-Recall:     {micro_r:.4f}")
    print(f"  Micro-F1:         {micro_f1:.4f}")
    print(f"\n  Macro-Precision:  {macro_p:.4f}")
    print(f"  Macro-Recall:     {macro_r:.4f}")
    print(f"  Macro-F1:         {macro_f1:.4f}")
    print(f"\n  Hamming Loss:     {h_loss:.4f}")
    print(f"  Subset Accuracy:  {subset_acc:.4f}")

    # ─── Per-label report ───────────────────────────────────
    print(f"\n  Per-Label Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_columns,
        zero_division=0,
    ))

    # ─── Score distribution per label ───────────────────────
    print(f"\n  Score Distribution (sigmoid probabilities):")
    for j, col in enumerate(label_columns):
        pos_mask = all_labels[:, j] == 1
        neg_mask = all_labels[:, j] == 0
        pos_scores = all_probs[pos_mask, j] if pos_mask.sum() > 0 else np.array([0])
        neg_scores = all_probs[neg_mask, j] if neg_mask.sum() > 0 else np.array([0])
        print(f"    {col:.<25s} pos_mean={pos_scores.mean():.3f}  neg_mean={neg_scores.mean():.3f}  "
              f"pos_count={int(pos_mask.sum())}")

    # ─── Overfitting analysis ──────────────────────────────
    history_path = os.path.join(output_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        print(f"\n  Train vs Val Loss Curve:")
        print(f"  {'Epoch':>5s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Gap':>10s}  {'Macro-F1':>10s}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for i, (tl, vl, mf) in enumerate(zip(
            history["train_loss"], history["val_loss"], history["macro_f1"]
        )):
            gap = vl - tl
            flag = " <-- OVERFITTING" if gap > 0.1 else ""
            print(f"  {i+1:>5d}  {tl:>10.4f}  {vl:>10.4f}  {gap:>+10.4f}  {mf:>10.4f}{flag}")

        final_gap = history["val_loss"][-1] - history["train_loss"][-1]
        if final_gap > 0.2:
            print(f"\n  WARNING: Large train-val gap ({final_gap:.3f}) suggests overfitting.")
        elif final_gap < 0.05:
            print(f"\n  OK: Train-val gap is small ({final_gap:.3f}) -- good generalization.")

    # ─── Save report ────────────────────────────────────────
    report = {
        "test_samples": len(all_labels),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc,
        "per_label_f1": f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        "label_columns": label_columns,
    }

    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model v2.0")
    parser.add_argument("--data_path", required=True, help="Path to dataset CSV/TSV")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory with trained model")
    parser.add_argument("--sep", default=None, help="CSV separator")
    parser.add_argument("--split", default=None, choices=['stratified', 'group', 'time'],
                        help="Split strategy: 'stratified' (default, with leakage), "
                             "'group' (0%% UA overlap, honest eval)")
    args = parser.parse_args()

    evaluate_model(args.data_path, args.output_dir, sep=args.sep, split_mode=args.split)


if __name__ == "__main__":
    main()

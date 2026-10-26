"""
Model v2.0 — Training Loop
Proper overfitting detection: tracks train AND val loss, early stopping, loss curves.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, WARMUP_RATIO, OUTPUT_DIR, CAPEC_LABELS, SIGMOID_THRESHOLD,
)
from dataset import create_dataloaders
from model import create_model


def compute_val_metrics(model, val_loader, device, threshold=SIGMOID_THRESHOLD):
    """Compute validation loss and per-label F1 scores."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    val_progress = tqdm(
        val_loader,
        desc="          [VAL]  ",
        leave=False,
        bar_format='{l_bar}{bar:30}{r_bar}',
        unit='batch',
    )

    with torch.no_grad():
        for batch in val_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            if loss.dim() > 0:  # DataParallel returns per-GPU losses
                loss = loss.mean()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(val_loader)

    # F1 scores
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        "val_loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label_f1": per_label_f1.tolist(),
    }


def train(data_path: str, output_dir: str = OUTPUT_DIR, epochs: int = NUM_EPOCHS,
          batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, sep=None,
          resume: bool = False):
    """Full training pipeline with overfitting safeguards."""

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model v2.0 — Training")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader, label_columns, class_weights = \
        create_dataloaders(data_path, batch_size=batch_size, sep=sep)

    # Save label columns mapping
    with open(os.path.join(output_dir, "label_columns.json"), "w") as f:
        json.dump(label_columns, f)

    # Create model
    num_labels = len(label_columns)
    model = create_model(num_labels=num_labels)
    model.to(device)
    class_weights = class_weights.to(device)

    # Multi-GPU support (e.g. Kaggle T4 x2)
    raw_model = model  # keep reference to unwrapped model
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    else:
        print(f"  Using single GPU")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision for GPU (fp16 — ~2x faster on T4/A100)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"  Mixed precision (fp16): {'ON' if use_amp else 'OFF'}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "micro_f1": [],
        "macro_f1": [],
        "per_label_f1": [],
        "best_epoch": 0,
        "best_macro_f1": 0.0,
        "label_columns": label_columns,
    }

    best_macro_f1 = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint if requested
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    if resume and os.path.exists(checkpoint_path):
        print(f"\n  Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_macro_f1 = ckpt["best_macro_f1"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        print(f"  Resumed from epoch {start_epoch} (best Macro-F1={best_macro_f1:.4f})")
    elif resume:
        print(f"  --resume specified but no checkpoint found at {checkpoint_path}, starting fresh.")

    print(f"\n  Starting training: epochs {start_epoch+1}-{epochs}, patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  {'='*60}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [TRAIN]",
            leave=True,
            bar_format='{l_bar}{bar:30}{r_bar}',
            unit='batch',
        )

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                loss = raw_model.get_weighted_loss(outputs["logits"], labels, class_weights)
                if loss.dim() > 0:  # DataParallel returns per-GPU losses
                    loss = loss.mean()

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            avg_so_far = train_loss / (step + 1)
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_so_far:.4f}",
                lr=f"{current_lr:.2e}",
            )

        avg_train_loss = train_loss / num_batches

        # Validation
        val_metrics = compute_val_metrics(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        # Log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["micro_f1"].append(val_metrics["micro_f1"])
        history["macro_f1"].append(val_metrics["macro_f1"])
        history["per_label_f1"].append(val_metrics["per_label_f1"])

        # Print epoch summary
        print(f"\n  Epoch {epoch+1}/{epochs} ({epoch_time:.0f}s)")
        print(f"    Train loss: {avg_train_loss:.4f}")
        print(f"    Val loss:   {val_metrics['val_loss']:.4f}")
        print(f"    Micro-F1:   {val_metrics['micro_f1']:.4f}")
        print(f"    Macro-F1:   {val_metrics['macro_f1']:.4f}")

        # Overfitting indicator
        if avg_train_loss < val_metrics["val_loss"] * 0.5:
            print(f"    ** WARNING: Train loss significantly lower than val loss — possible overfitting")

        # Early stopping check
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            history["best_epoch"] = epoch + 1
            history["best_macro_f1"] = best_macro_f1

            # Save best model
            torch.save(raw_model.state_dict(), os.path.join(output_dir, "model.pt"))
            print(f"    -> Saved best model (Macro-F1={best_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break

        # Save checkpoint for resume (every epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_macro_f1": best_macro_f1,
            "patience_counter": patience_counter,
            "history": history,
        }, checkpoint_path)
        print(f"    Checkpoint saved (can resume with --resume)")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final metrics
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best Macro-F1: {history['best_macro_f1']:.4f}")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*60}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Model v2.0")
    parser.add_argument("--data_path", required=True, help="Path to SR-BH 2020 CSV/TSV file")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sep", default=None, help="CSV separator (auto-detected if not set)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sep=args.sep,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

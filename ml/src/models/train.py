"""
ml/src/models/train.py
=======================
Full EfficientNetB0 training script with MLflow experiment tracking.
Two-phase training: frozen backbone → full fine-tune.

Every run logs:
  - Hyperparameters (lr, batch_size, epochs, dropout, frozen_layers)
  - Metrics per epoch (train_loss, val_loss, val_acc, val_f1)
  - Artefacts (best model weights, confusion matrix, class report)
  - Git commit hash (for DVC reproducibility requirement)
  - VRAM usage, training time per epoch

Usage:
    python ml/src/models/train.py
    python ml/src/models/train.py --batch_size 16 --epochs_phase2 20
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple


import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from ml.src.data.dataloader import (
    CLASS_NAMES,
    compute_class_weights,
    get_dataloaders,
)
from ml.src.models.model import get_model
from shared.config import get_config_value, resolve_path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]
PROC_DIR    = resolve_path(get_config_value("paths", "processed_data_dir", default="data/processed"))
MODELS_DIR  = resolve_path(get_config_value("paths", "models_dir", default="ml/models"))
EXP_DIR     = resolve_path(get_config_value("paths", "experiments_dir", default="ml/experiments"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = get_config_value(
    "ops",
    "mlflow",
    "tracking_uri",
    default="sqlite:///mlflow.db",
)
MLFLOW_EXPERIMENT_NAME = get_config_value(
    "ops",
    "mlflow",
    "experiment_name",
    default="radiologyai_xray_classification",
)
MODEL_ARCHITECTURE = get_config_value(
    "ml",
    "model",
    "architecture",
    default="EfficientNetB0",
)
MODEL_REGISTRY_NAME = get_config_value(
    "ml",
    "model",
    "registered_model_name",
    default="radiologyai_xray_classifier",
)
GRAD_CLIP_NORM = get_config_value("ml", "train", "grad_clip_norm", default=1.0)
WEIGHT_DECAY = get_config_value("ml", "train", "weight_decay", default=1e-4)
DATASET_NAME = get_config_value(
    "ml",
    "train",
    "dataset_name",
    default="chest-xray + covid19-radiography",
)
TRAIN_AUGMENTATION = get_config_value(
    "ml",
    "train",
    "augmentation",
    default="flip+brightness+rotation+noise",
)
IMAGE_SIZE = get_config_value("ml", "image_size", default=224)
MODEL_PATH = resolve_path(
    get_config_value("paths", "model_path", default="ml/models/efficientnetb0_best.pth")
)


def get_git_commit_hash() -> str:
    """
    Get current Git commit hash for MLflow reproducibility tag.

    Returns:
        Short commit hash string, or 'unknown' if not in a repo
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_vram_usage_mb() -> float:
    """
    Get current GPU VRAM usage in MB.

    Returns:
        VRAM allocated in MB, or 0.0 if no GPU
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def train_one_epoch(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    criterion  : nn.Module,
    optimizer  : torch.optim.Optimizer,
    device     : torch.device,
    epoch      : int,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model:     Neural network
        loader:    Training DataLoader
        criterion: Loss function
        optimizer: Optimiser
        device:    torch.device
        epoch:     Current epoch number (for logging)

    Returns:
        Dict with train_loss and train_acc
    """
    model.train()
    total_loss   = 0.0
    correct      = 0
    total        = 0
    start_time   = time.time()

    pbar = tqdm(loader, desc=f"  Epoch {epoch} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / total
    epoch_acc  = correct / total
    epoch_time = time.time() - start_time

    return {
        "train_loss" : round(epoch_loss, 4),
        "train_acc"  : round(epoch_acc,  4),
        "train_time" : round(epoch_time, 2),
    }


@torch.no_grad()
def evaluate(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    device    : torch.device,
    split     : str = "val",
) -> Dict[str, float]:
    """
    Evaluate the model on val or test set.

    Args:
        model:     Neural network
        loader:    DataLoader for val or test
        criterion: Loss function
        device:    torch.device
        split:     "val" or "test" (for logging label)

    Returns:
        Dict with loss, accuracy, macro F1
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n          = len(all_labels)
    epoch_loss = total_loss / n
    epoch_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        f"{split}_loss" : round(epoch_loss, 4),
        f"{split}_acc"  : round(epoch_acc,  4),
        f"{split}_f1"   : round(macro_f1,   4),
        "_preds"        : all_preds,
        "_labels"       : all_labels,
    }


def save_confusion_matrix(
    labels    : list,
    preds     : list,
    save_path : Path,
    title     : str = "Confusion Matrix",
) -> None:
    """
    Save confusion matrix as PNG artefact for MLflow.

    Args:
        labels:    True labels
        preds:     Predicted labels
        save_path: Output PNG path
        title:     Plot title
    """
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def train(args: argparse.Namespace) -> None:
    """
    Main training function — two-phase training with MLflow tracking.

    Phase 1: Frozen backbone, train classifier head only
    Phase 2: Unfreeze all layers, fine-tune end-to-end

    Args:
        args: Parsed command-line arguments
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit     = get_git_commit_hash()

    logger.info("=" * 60)
    logger.info("RadiologyAI — EfficientNetB0 Training")
    logger.info("=" * 60)
    logger.info(f"Device       : {device}")
    logger.info(f"Git commit   : {commit}")
    logger.info(f"Batch size   : {args.batch_size}")
    logger.info(f"Phase 1 epochs: {args.epochs_phase1}")
    logger.info(f"Phase 2 epochs: {args.epochs_phase2}")

    # ── Data ──────────────────────────────────────────────────
    loaders      = get_dataloaders(PROC_DIR, batch_size=args.batch_size)
    class_weights = compute_class_weights(PROC_DIR, device)

    # ── MLflow setup ──────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{MODEL_ARCHITECTURE.lower()}_{commit}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # ── Log hyperparameters ───────────────────────────────
        mlflow.log_params({
            "model"         : MODEL_ARCHITECTURE,
            "framework"     : "PyTorch",
            "num_classes"   : len(CLASS_NAMES),
            "image_size"    : IMAGE_SIZE,
            "batch_size"    : args.batch_size,
            "epochs_phase1" : args.epochs_phase1,
            "epochs_phase2" : args.epochs_phase2,
            "lr_phase1"     : args.lr_phase1,
            "lr_phase2"     : args.lr_phase2,
            "dropout"       : args.dropout,
            "optimizer"     : "AdamW",
            "scheduler"     : "CosineAnnealingLR",
            "augmentation"  : TRAIN_AUGMENTATION,
            "class_weights" : "balanced",
            "dataset"       : DATASET_NAME,
            "git_commit"    : commit,           # reproducibility requirement
        })

        # manually log dataset stats (beyond autolog)
        split_stats_path = PROC_DIR / "split_stats.json"
        if split_stats_path.exists():
            with open(split_stats_path) as f:
                split_stats = json.load(f)
            mlflow.log_dict(split_stats, "dataset_split_stats.json")

        # ── Model & loss ──────────────────────────────────────
        model     = get_model(device, freeze=True, dropout=args.dropout)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1  = 0.0
        best_model_path = MODEL_PATH

        # ══════════════════════════════════════════════════════
        # PHASE 1 — Frozen backbone, train head only
        # ══════════════════════════════════════════════════════
        logger.info("\nPHASE 1 — Training classifier head (backbone frozen)")
        optimizer1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_phase1, weight_decay=WEIGHT_DECAY
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=args.epochs_phase1
        )

        for epoch in range(1, args.epochs_phase1 + 1):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model, loaders["train"], criterion, optimizer1, device, epoch
            )
            val_metrics = evaluate(
                model, loaders["val"], criterion, device, split="val"
            )
            scheduler1.step()

            epoch_time = time.time() - epoch_start
            vram_mb    = get_vram_usage_mb()

            # log to MLflow
            metrics = {
                **train_metrics,
                "val_loss"  : val_metrics["val_loss"],
                "val_acc"   : val_metrics["val_acc"],
                "val_f1"    : val_metrics["val_f1"],
                "lr"        : scheduler1.get_last_lr()[0],
                "vram_mb"   : round(vram_mb, 1),
                "epoch_time": round(epoch_time, 1),
            }
            mlflow.log_metrics(metrics, step=epoch)

            logger.info(
                f"  P1 Epoch {epoch:02d}/{args.epochs_phase1} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | "
                f"val_f1={val_metrics['val_f1']:.4f} | "
                f"vram={vram_mb:.0f}MB"
            )

            # save best model
            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  Best model saved (val_f1={best_val_f1:.4f})")

        # ══════════════════════════════════════════════════════
        # PHASE 2 — Unfreeze all, fine-tune end-to-end
        # ══════════════════════════════════════════════════════
        logger.info("\nPHASE 2 — Full fine-tuning (backbone unfrozen)")
        model.unfreeze_backbone()

        optimizer2 = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_phase2, weight_decay=WEIGHT_DECAY
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=args.epochs_phase2
        )

        total_epochs = args.epochs_phase1 + args.epochs_phase2

        for epoch in range(args.epochs_phase1 + 1, total_epochs + 1):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model, loaders["train"], criterion, optimizer2, device, epoch
            )
            val_metrics = evaluate(
                model, loaders["val"], criterion, device, split="val"
            )
            scheduler2.step()

            epoch_time = time.time() - epoch_start
            vram_mb    = get_vram_usage_mb()

            metrics = {
                **train_metrics,
                "val_loss"  : val_metrics["val_loss"],
                "val_acc"   : val_metrics["val_acc"],
                "val_f1"    : val_metrics["val_f1"],
                "lr"        : scheduler2.get_last_lr()[0],
                "vram_mb"   : round(vram_mb, 1),
                "epoch_time": round(epoch_time, 1),
            }
            mlflow.log_metrics(metrics, step=epoch)

            logger.info(
                f"  P2 Epoch {epoch:02d}/{total_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | "
                f"val_f1={val_metrics['val_f1']:.4f} | "
                f"vram={vram_mb:.0f}MB"
            )

            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  Best model saved (val_f1={best_val_f1:.4f})")

        # ══════════════════════════════════════════════════════
        # FINAL EVALUATION on test set
        # ══════════════════════════════════════════════════════
        logger.info("\nFINAL TEST SET EVALUATION")
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = evaluate(
            model, loaders["test"], criterion, device, split="test"
        )

        mlflow.log_metrics({
            "test_loss" : test_metrics["test_loss"],
            "test_acc"  : test_metrics["test_acc"],
            "test_f1"   : test_metrics["test_f1"],
            "best_val_f1": best_val_f1,
        })

        # classification report
        report = classification_report(
            test_metrics["_labels"],
            test_metrics["_preds"],
            target_names=CLASS_NAMES,
            output_dict=True,
        )
        mlflow.log_dict(report, "classification_report.json")

        # confusion matrix artefact
        cm_path = EXP_DIR / "confusion_matrix.png"
        save_confusion_matrix(
            test_metrics["_labels"],
            test_metrics["_preds"],
            cm_path,
        )
        mlflow.log_artifact(str(cm_path))

        # log model to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path   = "model",
            registered_model_name = MODEL_REGISTRY_NAME,
        )

        # save test metrics to file (DVC metrics)
        test_metrics_clean = {
            k: v for k, v in test_metrics.items()
            if not k.startswith("_")
        }
        metrics_path = EXP_DIR / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics_clean, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Best val F1  : {best_val_f1:.4f}")
        logger.info(f"  Test acc     : {test_metrics['test_acc']:.4f}")
        logger.info(f"  Test F1      : {test_metrics['test_f1']:.4f}")
        logger.info(f"  MLflow run   : {run_id}")
        logger.info(f"  Best weights : {best_model_path}")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="RadiologyAI Training")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=get_config_value("ml", "train", "batch_size", default=32),
    )
    parser.add_argument(
        "--epochs_phase1",
        type=int,
        default=get_config_value("ml", "train", "epochs_phase1", default=5),
    )
    parser.add_argument(
        "--epochs_phase2",
        type=int,
        default=get_config_value("ml", "train", "epochs_phase2", default=15),
    )
    parser.add_argument(
        "--lr_phase1",
        type=float,
        default=get_config_value("ml", "train", "lr_phase1", default=1e-3),
    )
    parser.add_argument(
        "--lr_phase2",
        type=float,
        default=get_config_value("ml", "train", "lr_phase2", default=1e-4),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=get_config_value("ml", "train", "dropout", default=0.3),
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=MLFLOW_TRACKING_URI,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

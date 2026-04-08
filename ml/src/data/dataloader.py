"""
ml/src/data/dataloader.py
==========================
PyTorch Dataset + DataLoader for chest X-ray classification.
Replaces the tf.data pipeline — GPU-pinned memory, augmentation
via albumentations, class-weight computation built in.

Classes:
    XRayDataset   — torch.utils.data.Dataset for one split
    get_dataloaders — returns train/val/test DataLoader objects
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
IMAGE_SIZE  = 224
CLASS_NAMES = ["Normal", "Pneumonia", "COVID19"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ── Albumentations transforms ─────────────────────────────────
def get_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline.
    Clinically safe augmentations only — no vertical flip
    (inverting lungs is not anatomically valid).

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],    # ImageNet mean
            std=[0.229, 0.224, 0.225],     # ImageNet std
        ),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Validation / test transforms — resize + normalise only.
    No augmentation to ensure reproducible evaluation.

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────
class XRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.

    Folder structure expected:
        processed_dir/
            split/
                Normal/
                Pneumonia/
                COVID19/

    Args:
        processed_dir: Root path to data/processed/
        split:         "train", "val", or "test"
        transform:     Albumentations transform pipeline
    """

    def __init__(
        self,
        processed_dir : Path,
        split         : str,
        transform     : Optional[A.Compose] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.split         = split
        self.transform     = transform
        self.samples       : List[Tuple[Path, int]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Scan class folders and build (path, label) list."""
        split_dir = self.processed_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Run preprocess.py first."
            )

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class folder missing: {class_dir}")
                continue

            images = (
                list(class_dir.glob("*.jpg"))   +
                list(class_dir.glob("*.jpeg"))  +
                list(class_dir.glob("*.png"))
            )

            for img_path in images:
                self.samples.append((img_path, class_idx))

            logger.info(
                f"  {self.split}/{class_name}: {len(images)} images"
            )

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

        logger.info(
            f"Dataset [{self.split}]: {len(self.samples)} total samples"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and transform a single sample.

        Args:
            idx: Sample index

        Returns:
            (image_tensor [C,H,W], class_index)
        """
        img_path, label = self.samples[idx]

        # Load as RGB numpy array (albumentations expects numpy)
        image = np.array(
            Image.open(img_path).convert("RGB"),
            dtype=np.uint8
        )

        if self.transform:
            augmented = self.transform(image=image)
            image     = augmented["image"]          # torch.Tensor [C,H,W]

        return image, label


# ── DataLoader factory ────────────────────────────────────────
def get_dataloaders(
    processed_dir : Path,
    batch_size    : int  = 32,
    num_workers   : int  = 4,
    pin_memory    : bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Args:
        processed_dir: Path to data/processed/
        batch_size:    Images per batch
        num_workers:   CPU workers for data loading
        pin_memory:    Pin memory for faster GPU transfer

    Returns:
        Dict with keys "train", "val", "test"
    """
    datasets = {
        "train": XRayDataset(
            processed_dir,
            split     = "train",
            transform = get_train_transforms(),
        ),
        "val": XRayDataset(
            processed_dir,
            split     = "val",
            transform = get_val_transforms(),
        ),
        "test": XRayDataset(
            processed_dir,
            split     = "test",
            transform = get_val_transforms(),
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
    }

    return loaders


def compute_class_weights(
    processed_dir: Path,
    device       : torch.device,
) -> torch.Tensor:
    """
    Compute per-class weights for weighted cross-entropy loss.
    Formula: weight_i = total / (n_classes * count_i)

    Args:
        processed_dir: Path to data/processed/
        device:        torch.device to move weights to

    Returns:
        Tensor of shape [n_classes] on the given device
    """
    counts = []
    train_dir = Path(processed_dir) / "train"

    for class_name in CLASS_NAMES:
        class_dir = train_dir / class_name
        if class_dir.exists():
            n = len(list(class_dir.glob("*.jpg")))
            counts.append(n)
        else:
            counts.append(1)           # avoid division by zero

    total  = sum(counts)
    n_cls  = len(counts)
    weights = [total / (n_cls * c) for c in counts]

    logger.info("Class weights:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        logger.info(f"  {name:<12} count={counts[i]:>5}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ROOT     = Path(__file__).resolve().parents[3]
    PROC_DIR = ROOT / "data" / "processed"

    loaders = get_dataloaders(PROC_DIR, batch_size=32)

    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"\n[{split}]")
        print(f"  Batch shape : {images.shape}")
        print(f"  Labels      : {labels[:8].tolist()}")
        print(f"  Image range : [{images.min():.3f}, {images.max():.3f}]")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = compute_class_weights(PROC_DIR, device)
    print(f"\nClass weights tensor: {weights}")
    print(f"Device: {device}")
    print("\nDataLoader test passed!")

"""
ml/src/models/model.py
=======================
EfficientNetB0 fine-tuned for 3-class chest X-ray classification.
Uses torchvision pretrained weights (ImageNet).

Two-phase training strategy:
  Phase 1 — Frozen backbone, train classifier head only (5 epochs)
  Phase 2 — Unfreeze all layers, fine-tune end-to-end (15 epochs)

Classes:
    XRayClassifier — the full model wrapper
    get_model      — factory function used by train.py
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from shared.config import get_config_value

logger = logging.getLogger(__name__)

CLASS_NAMES = get_config_value(
    "ml",
    "class_names",
    default=["Normal", "Pneumonia", "COVID19"],
)
NUM_CLASSES = len(CLASS_NAMES)


class XRayClassifier(nn.Module):
    """
    EfficientNetB0 with a custom classification head
    for 3-class chest X-ray diagnosis.

    Architecture:
        EfficientNetB0 backbone (ImageNet pretrained)
        → Dropout(0.3)
        → Linear(1280, 256)
        → ReLU
        → Dropout(0.2)
        → Linear(256, 3)

    Args:
        num_classes: Number of output classes (default 3)
        dropout:     Dropout rate before final layer
        freeze:      If True, freeze backbone on init (Phase 1)
    """

    def __init__(
        self,
        num_classes : int   = NUM_CLASSES,
        dropout     : float = 0.3,
        freeze      : bool  = True,
    ):
        super().__init__()

        # ── Load pretrained EfficientNetB0 ────────────────────
        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # ── Replace classifier head ───────────────────────────
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(256, num_classes),
        )

        if freeze:
            self.freeze_backbone()

        logger.info(
            f"XRayClassifier initialised — "
            f"backbone={'frozen' if freeze else 'unfrozen'}, "
            f"num_classes={num_classes}, dropout={dropout}"
        )

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.
        Used for Phase 1 training — only the classifier head updates.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will train.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters.
        Used for Phase 2 fine-tuning — full model updates.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full model fine-tuning active.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Logits tensor [B, num_classes]
        """
        return self.backbone(x)

    def count_trainable_params(self) -> Tuple[int, int]:
        """
        Count trainable vs total parameters.

        Returns:
            (trainable_params, total_params)
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


def get_model(
    device  : torch.device,
    freeze  : bool  = True,
    dropout : float = 0.3,
) -> XRayClassifier:
    """
    Build and move the model to the target device.

    Args:
        device:  torch.device (cuda or cpu)
        freeze:  Freeze backbone initially
        dropout: Dropout rate

    Returns:
        XRayClassifier on the given device
    """
    model = XRayClassifier(
        num_classes = NUM_CLASSES,
        dropout     = dropout,
        freeze      = freeze,
    )
    model = model.to(device)

    trainable, total = model.count_trainable_params()
    logger.info(
        f"Model on {device} | "
        f"trainable={trainable:,} / total={total:,} params "
        f"({100*trainable/total:.1f}%)"
    )
    return model


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(device, freeze=True)

    # dummy forward pass
    x      = torch.randn(4, 3, 224, 224).to(device)
    logits = model(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Output sample: {logits[0].detach().cpu().tolist()}")
    print("Model test passed!")

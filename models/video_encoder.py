"""
models/video_encoder.py
-----------------------
Visual feature extractor using ResNet-50 (pretrained on ImageNet).

Input:  (B, T, C, H, W)  — batch of frame sequences
Output: (B, T, D)         — per-frame feature vectors (D = 2048 for ResNet-50)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class VideoEncoder(nn.Module):
    """
    Encodes a sequence of face-cropped frames into a feature sequence
    using a pretrained ResNet backbone.

    The same ResNet is applied independently to each frame (weight-shared),
    yielding a temporal sequence of spatial features.

    Args:
        backbone (str): 'resnet50' | 'resnet18'
        pretrained (bool): Load ImageNet weights.
        freeze (bool): Freeze all backbone weights (unfreeze later for fine-tuning).
        feature_dim (int): Output feature dimension (2048 for R50, 512 for R18).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = 2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Load backbone ──────────────────────────────────────────────────
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            assert feature_dim == 2048, "ResNet-50 outputs 2048-d features."
        elif backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            assert feature_dim == 512, "ResNet-18 outputs 512-d features."
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet50' or 'resnet18'.")

        # Remove the final FC layer — keep up to global average pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B, D, 1, 1)

        if freeze:
            self.freeze_backbone()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)  — T frames per video

        Returns:
            features: (B, T, D)  — per-frame feature vectors
        """
        B, T, C, H, W = frames.shape

        # Merge batch & time dims for efficient batch processing
        x = frames.view(B * T, C, H, W)         # (B*T, C, H, W)
        x = self.backbone(x)                      # (B*T, D, 1, 1)
        x = x.flatten(1)                          # (B*T, D)

        # Restore temporal dimension
        features = x.view(B, T, self.feature_dim) # (B, T, D)
        return features

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

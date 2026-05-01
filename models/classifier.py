"""
models/classifier.py
--------------------
MLP classification head for binary deepfake detection.

Input:  (B, D)  — pooled fused representation
Output: (B, 1)  — logit (apply sigmoid for probability)
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Two-layer MLP with dropout for binary classification.

    Args:
        input_dim (int): Dimension of the fused feature vector.
        hidden_dim (int): Hidden layer size.
        dropout (float): Dropout rate before each linear layer.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   # Raw logit; apply sigmoid externally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D)

        Returns:
            logits: (B, 1)
        """
        return self.net(x)

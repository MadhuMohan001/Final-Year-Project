"""
models/deepfake_model.py
------------------------
Full deepfake detection model.

Pipeline:
    frames  (B, T, C, H, W) → VideoEncoder  → (B, T_v, D_v)
                                             → Linear proj → (B, T_v, D)
    audio   (B, L)          → AudioEncoder  → (B, T_a, D_a)
                                             → Linear proj → (B, T_a, D)
    (B, T_a, D) + (B, T_v, D) → CrossAttentionFusion → (B, T_a, D)
                                                       → mean pool → (B, D)
                                                       → MLPClassifier → (B, 1)
"""

import torch
import torch.nn as nn

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.cross_attention import CrossAttentionFusion
from models.classifier import MLPClassifier


class DeepfakeDetector(nn.Module):
    """
    End-to-end multimodal deepfake detection with cross-attention fusion.

    Args:
        cfg (dict): Full model config (from config.yaml['model']).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        proj_dim      = m["projection_dim"]       # shared latent dim
        video_dim     = m["video_feature_dim"]    # ResNet output dim
        audio_dim     = m["audio_feature_dim"]    # Wav2Vec output dim
        num_heads     = m["num_heads"]
        num_layers    = m["num_attn_layers"]
        dropout       = m["dropout"]
        hidden_dim    = m["hidden_dim"]
        bidirectional = m.get("bidirectional_attention", False)

        # ── Encoders ──────────────────────────────────────────────────────
        self.video_encoder = VideoEncoder(
            backbone    = m["video_encoder"],
            pretrained  = m["video_pretrained"],
            freeze      = cfg["train"]["freeze_video_encoder"],
            feature_dim = video_dim,
        )

        self.audio_encoder = AudioEncoder(
            model_name  = m["audio_encoder"],
            pretrained  = m["audio_pretrained"],
            freeze      = cfg["train"]["freeze_audio_encoder"],
            feature_dim = audio_dim,
        )

        # ── Projection layers (map both modalities to shared dim) ─────────
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── Cross-Attention Fusion ────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            d_model       = proj_dim,
            num_heads     = num_heads,
            num_layers    = num_layers,
            dropout       = dropout,
            bidirectional = bidirectional,
        )

        # ── MLP Classifier ────────────────────────────────────────────────
        self.classifier = MLPClassifier(
            input_dim  = proj_dim,
            hidden_dim = hidden_dim,
            dropout    = dropout,
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,    # (B, T, C, H, W)
        audio:  torch.Tensor,    # (B, L)
    ) -> dict:
        """
        Returns:
            dict with:
                'logits' : (B, 1)   — raw logits
                'probs'  : (B, 1)   — sigmoid probabilities
        """
        # ── Encode ────────────────────────────────────────────────────────
        V = self.video_encoder(frames)   # (B, T_v, D_v)
        A = self.audio_encoder(audio)    # (B, T_a, D_a)

        # ── Project to shared space ───────────────────────────────────────
        V = self.video_proj(V)           # (B, T_v, D)
        A = self.audio_proj(A)           # (B, T_a, D)

        # ── Fuse via cross-attention ──────────────────────────────────────
        F = self.fusion(A, V)            # (B, T_a, D)

        # ── Temporal pooling: mean over audio timesteps ───────────────────
        F_pooled = F.mean(dim=1)         # (B, D)

        # ── Classify ──────────────────────────────────────────────────────
        logits = self.classifier(F_pooled)  # (B, 1)
        probs  = torch.sigmoid(logits)

        return {"logits": logits, "probs": probs}

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────

    def unfreeze_encoders(self):
        """Unfreeze both backbone encoders for full fine-tuning."""
        self.video_encoder.unfreeze_backbone()
        self.audio_encoder.unfreeze_backbone()
        print("[Model] Encoders unfrozen for fine-tuning.")

    def get_attention_weights(self, layer_idx: int = -1):
        """Retrieve cross-attention weights (B, T_a, T_v) for the last forward pass."""
        return self.fusion.get_attention_weights(layer_idx=layer_idx)

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params:     {total:,}")
        print(f"[Model] Trainable params: {trainable:,}")
        return trainable

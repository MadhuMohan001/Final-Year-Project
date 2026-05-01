"""
models/cross_attention.py
--------------------------
Multi-head Cross-Attention Fusion module.

Given:
    A  (B, T_a, D)  — audio feature sequence  (Query)
    V  (B, T_v, D)  — video feature sequence  (Key & Value)

Learns which video frames align with each audio timestep.
Outputs a fused sequence that encodes audio-visual consistency.

Inconsistencies between audio and video (common in deepfakes)
show up as diffuse, low-confidence attention weight distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer.

    Query  = one modality (audio A)
    Key    = other modality (video V)
    Value  = other modality (video V)

    Followed by a position-wise FFN and layer-norm residual connections.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.scale     = self.d_head ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Store last attention weights for visualization
        self.last_attn_weights = None   # (B, H, T_q, T_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, D)
            key:   (B, T_k, D)
            value: (B, T_k, D)

        Returns:
            out: (B, T_q, D)  — attended query features
        """
        B, T_q, D = query.shape
        T_k = key.shape[1]

        # ── Multi-head attention ──────────────────────────────────────────
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, T_q, d)
        K = self.k_proj(key  ).view(B, T_k, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, T_k, d)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, T_k, d)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, T_q, T_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        self.last_attn_weights = attn_weights.detach().cpu()  # save for visualization

        attended = torch.matmul(attn_weights, V)                           # (B, H, T_q, d)
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, D)  # (B, T_q, D)
        attended = self.out_proj(attended)

        # ── Residual + LayerNorm ──────────────────────────────────────────
        query = self.norm1(query + self.dropout(attended))

        # ── FFN ───────────────────────────────────────────────────────────
        query = self.norm2(query + self.dropout(self.ffn(query)))

        return query   # (B, T_q, D)


class CrossAttentionFusion(nn.Module):
    """
    Stacked cross-attention fusion for audio-visual deepfake detection.

    Supports:
      - Unidirectional: Audio attends to Video (default)
      - Bidirectional:  Audio attends to Video AND Video attends to Audio,
                        then both outputs are concatenated + projected.

    Args:
        d_model (int): Projected feature dimension (both modalities share this).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of stacked cross-attention layers.
        dropout (float): Dropout rate.
        bidirectional (bool): Enable bidirectional cross-attention.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # Audio-attends-to-Video layers
        self.a2v_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        if bidirectional:
            # Video-attends-to-Audio layers
            self.v2a_layers = nn.ModuleList([
                CrossAttentionLayer(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ])
            # Project concatenated output back to d_model
            self.bi_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_feat: (B, T_a, D)  — projected audio features
            video_feat: (B, T_v, D)  — projected video features

        Returns:
            fused: (B, T_q, D)  — fused representation
                   T_q = T_a for unidirectional, T_a for bidirectional (audio side)
        """
        # ── Audio → Video attention ───────────────────────────────────────
        a = audio_feat
        for layer in self.a2v_layers:
            a = layer(query=a, key=video_feat, value=video_feat)
        # a: (B, T_a, D)

        if not self.bidirectional:
            return a

        # ── Video → Audio attention (bidirectional) ────────────────────────
        v = video_feat
        for layer in self.v2a_layers:
            v = layer(query=v, key=audio_feat, value=audio_feat)
        # v: (B, T_v, D) — pool to match audio length
        v_pooled = v.mean(dim=1, keepdim=True).expand_as(a)  # (B, T_a, D)

        # Concatenate and project
        fused = self.bi_proj(torch.cat([a, v_pooled], dim=-1))  # (B, T_a, D)
        return fused

    def get_attention_weights(self, layer_idx: int = -1) -> torch.Tensor:
        """
        Retrieve saved attention weights for visualization.

        Returns:
            Tensor (B, H, T_a, T_v) averaged over heads → (B, T_a, T_v)
        """
        layer = self.a2v_layers[layer_idx]
        if layer.last_attn_weights is None:
            return None
        return layer.last_attn_weights.mean(dim=1)  # avg over heads → (B, T_a, T_v)

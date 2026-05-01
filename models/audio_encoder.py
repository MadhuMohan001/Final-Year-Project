"""
models/audio_encoder.py
-----------------------
Audio feature extractor using pretrained Wav2Vec 2.0 (HuggingFace transformers).

Input:  (B, L)      — raw waveform (16 kHz, float32, normalised)
Output: (B, T_a, D) — contextual audio representations (D = 768 for base model)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config


class AudioEncoder(nn.Module):
    """
    Encodes raw audio waveforms into contextual feature sequences using
    the pretrained Wav2Vec 2.0 Transformer.

    Args:
        model_name (str): HuggingFace model ID, e.g. 'facebook/wav2vec2-base'.
        pretrained (bool): Load pretrained weights.
        freeze (bool): Freeze all Wav2Vec weights initially.
        feature_dim (int): Output hidden size (768 for base, 1024 for large).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = 768,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Load Wav2Vec 2.0 ──────────────────────────────────────────────
        if pretrained:
            self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        else:
            config = Wav2Vec2Config()
            self.wav2vec = Wav2Vec2Model(config)

        if freeze:
            self.freeze_backbone()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, L)  — raw audio at 16 kHz

        Returns:
            features: (B, T_a, D)  — contextual audio feature sequence
                      T_a depends on audio length (≈L / 320 for base model)
        """
        # Wav2Vec 2.0 expects (B, L) float32 tensors
        outputs = self.wav2vec(input_values=waveforms, return_dict=True)

        # last_hidden_state: (B, T_a, D)
        return outputs.last_hidden_state

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────

    def freeze_backbone(self):
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.wav2vec.parameters():
            param.requires_grad = True

    def freeze_feature_extractor_only(self):
        """
        Partial freeze strategy: freeze only the CNN feature extractor
        (first stage of Wav2Vec), keep the Transformer layers trainable.
        Useful for cheaper fine-tuning.
        """
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.wav2vec.encoder.parameters():
            param.requires_grad = True

"""
inference.py
------------
Run deepfake detection on a single video file.

Usage:
    python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth
    python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth --visualize
"""

import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.deepfake_model import DeepfakeDetector
from utils.face_detector import FaceDetector
from utils.audio_utils import extract_audio_from_video, preprocess_audio


DEFAULT_CONFIG = "configs/config.yaml"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_video_transform(frame_size: int = 224):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


def predict(video_path: str, checkpoint_path: str, cfg: dict, visualize: bool = False) -> dict:
    """
    Run inference on a single video.

    Returns:
        dict:
            'label'       : str  — 'REAL' or 'FAKE'
            'probability' : float — probability of being FAKE
            'confidence'  : float — max(prob, 1-prob)
    """
    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    data_cfg  = cfg["data"]
    model_cfg = cfg

    # ── Load model ────────────────────────────────────────────────────────
    model = DeepfakeDetector(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Extract video frames ──────────────────────────────────────────────
    face_detector = FaceDetector(
        image_size=data_cfg["frame_size"],
        device=str(device),
    )
    face_crops = face_detector.extract_faces_from_video(
        video_path,
        num_frames=data_cfg["num_frames"],
        fps=data_cfg.get("fps"),
    )

    transform = build_video_transform(data_cfg["frame_size"])
    frame_tensors = [transform(Image.fromarray(f)) for f in face_crops]
    frames = torch.stack(frame_tensors).unsqueeze(0).to(device)   # (1, T, C, H, W)

    # ── Extract audio ─────────────────────────────────────────────────────
    waveform = extract_audio_from_video(video_path, target_sr=data_cfg["sample_rate"])
    waveform = preprocess_audio(waveform, max_len=data_cfg["max_audio_len"])
    audio    = torch.from_numpy(waveform).unsqueeze(0).to(device)  # (1, L)

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out  = model(frames, audio)
        prob = out["probs"].squeeze().item()

    threshold  = cfg["inference"]["threshold"]
    is_fake    = prob >= threshold
    label      = "FAKE" if is_fake else "REAL"
    confidence = max(prob, 1 - prob)

    result = {
        "label":       label,
        "probability": prob,
        "confidence":  confidence,
    }

    # ── Print result ──────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  Video     : {os.path.basename(video_path)}")
    print(f"  Prediction: {label}")
    print(f"  Fake prob : {prob:.4f}")
    print(f"  Confidence: {confidence:.4f}")
    print("─" * 50 + "\n")

    # ── Attention visualization ───────────────────────────────────────────
    if visualize and cfg["inference"].get("visualize_attention", True):
        attn = model.get_attention_weights(layer_idx=-1)  # (1, T_a, T_v)
        if attn is not None:
            from utils.visualize import plot_attention_map
            attn_map = attn[0].numpy()   # (T_a, T_v)
            save_path = os.path.join("outputs", f"{os.path.splitext(os.path.basename(video_path))[0]}_attention.png")
            os.makedirs("outputs", exist_ok=True)
            plot_attention_map(
                attn_map,
                title=f"Cross-Attention — {label} (p={prob:.3f})",
                save_path=save_path,
            )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, required=True,         help="Path to input video")
    parser.add_argument("--checkpoint", type=str, required=True,         help="Path to model checkpoint")
    parser.add_argument("--config",     type=str, default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--visualize",  action="store_true",             help="Plot cross-attention map")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    result = predict(args.video, args.checkpoint, cfg, visualize=args.visualize)

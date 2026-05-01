"""
evaluate.py
-----------
Evaluate a trained deepfake detector on the test split.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.deepfake_model import DeepfakeDetector
from utils.metrics import compute_metrics, print_metrics, get_confusion_matrix, get_classification_report
from utils.visualize import plot_confusion_matrix


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(cfg: dict, checkpoint_path: str):
    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f"[Eval] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(cfg, device=str(device))
    print(f"[Eval] Test batches: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = DeepfakeDetector(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}")
    if "val_metrics" in ckpt:
        print(f"[Eval] Saved val F1 at checkpoint: {ckpt['val_metrics'].get('f1', 'N/A'):.4f}")

    threshold = cfg["eval"]["threshold"]

    # ── Inference ─────────────────────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            frames = batch["frames"].to(device)
            audio  = batch["audio" ].to(device)
            labels = batch["label" ].cpu().numpy().astype(int)

            out   = model(frames, audio)
            probs = out["probs"].squeeze(1).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n" + "─" * 60)
    print_metrics(metrics, prefix="Test Set")
    print("─" * 60)
    print(get_classification_report(y_true, y_pred))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = get_confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    save_path = os.path.join("outputs", "confusion_matrix.png")
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(cm, save_path=save_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)

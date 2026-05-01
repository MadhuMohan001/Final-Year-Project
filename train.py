"""
train.py
--------
Full training loop for the deepfake detection model.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/epoch_10.pth
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.deepfake_model import DeepfakeDetector
from utils.metrics import compute_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["train"]
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg: dict, num_train_steps: int):
    train_cfg = cfg["train"]
    scheduler_type = train_cfg.get("scheduler", "cosine")
    warmup_steps   = train_cfg["warmup_epochs"] * num_train_steps

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=train_cfg["learning_rate"],
            total_steps=train_cfg["epochs"] * num_train_steps,
            pct_start=warmup_steps / (train_cfg["epochs"] * num_train_steps),
            anneal_strategy="cos",
            final_div_factor=train_cfg["learning_rate"] / train_cfg.get("min_lr", 1e-6),
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool, log_every: int = 10):
    model.train(train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, desc="Train" if train else "Val  ", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        frames = batch["frames"].to(device)   # (B, T, C, H, W)
        audio  = batch["audio" ].to(device)   # (B, L)
        labels = batch["label" ].to(device)   # (B,)

        with torch.set_grad_enabled(train):
            out    = model(frames, audio)
            logits = out["logits"].squeeze(1)            # (B,)
            loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labs  = labels.detach().cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labs.tolist())

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    metrics["loss"] = avg_loss
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str = None):
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    print(f"[Train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("[Train] Building dataloaders …")
    train_loader, val_loader, _ = build_dataloaders(cfg, device=str(device))
    print(f"[Train] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("[Train] Building model …")
    model = DeepfakeDetector(cfg).to(device)
    model.count_parameters()

    # ── Loss ──────────────────────────────────────────────────────────────
    pos_weight = torch.tensor([cfg["train"].get("pos_weight", 1.0)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg, num_train_steps=len(train_loader))

    # ── Optionally resume ─────────────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_f1  = ckpt.get("best_val_f1", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}")

    # ── Logging ───────────────────────────────────────────────────────────
    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["train"]["log_dir"])

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # ── Epoch loop ────────────────────────────────────────────────────────
    epochs = cfg["train"]["epochs"]
    unfreeze_epoch = cfg["train"].get("unfreeze_epoch", epochs + 1)

    for epoch in range(start_epoch, epochs):
        print(f"\n── Epoch {epoch + 1}/{epochs} ──────────────────────────────")

        # Unfreeze backbones at the specified epoch
        if epoch == unfreeze_epoch:
            model.unfreeze_encoders()
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, num_train_steps=len(train_loader))
            print("[Train] Encoders unfrozen — rebuilding optimizer.")

        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, train=True, log_every=cfg["train"]["log_every"],
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, None, None,
            device, train=False,
        )
        elapsed = time.time() - t0

        # Console
        print_metrics(train_metrics, prefix=f"Train Ep{epoch+1}")
        print_metrics(val_metrics,   prefix=f"Val   Ep{epoch+1}")
        print(f"  Epoch time: {elapsed:.1f}s")

        # TensorBoard
        for k in ["loss", "accuracy", "f1", "auc_roc"]:
            writer.add_scalars(k, {"train": train_metrics[k], "val": val_metrics[k]}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])

        # ── Checkpointing ──────────────────────────────────────────────────
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_val_f1": best_val_f1,
            "cfg":         cfg,
        }

        if cfg["train"]["save_best"] and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            ckpt["best_val_f1"] = best_val_f1
            best_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth")
            torch.save(ckpt, best_path)
            print(f"  ✓ Best model saved (val F1 = {best_val_f1:.4f})")

        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ep_path = os.path.join(cfg["train"]["checkpoint_dir"], f"epoch_{epoch+1:03d}.pth")
            torch.save(ckpt, ep_path)

    writer.close()
    print("\n[Train] Training complete.")
    print(f"[Train] Best val F1: {best_val_f1:.4f}")

    # ── Plot training curves ───────────────────────────────────────────────
    from utils.visualize import plot_training_curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(cfg["train"]["log_dir"], "training_curves.png"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)

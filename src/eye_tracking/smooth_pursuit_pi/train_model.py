#!/usr/bin/env python3
"""
Training script for the Pi-based smooth pursuit gaze model.
Reads prepared dataset (from prepare_dataset.py) and trains MobileNetV2.

Usage:
    python train_model.py --dataset_dir ./dataset --distance 40
    python train_model.py --dataset_dir ./dataset --distance 40 --epochs 50 --lr 0.001
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobilenet.model import GazeMobileNet


class PiSmoothPursuitDataset(Dataset):
    """Dataset from prepared Pi session data (labels.csv + frame images)."""

    def __init__(self, labels: list[dict], dataset_dir: Path, transform=None):
        self.labels = labels
        self.dataset_dir = dataset_dir
        self.transform = transform or self._default_transform()

    @staticmethod
    def _default_transform(augment=False):
        if augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        frame_path = self.dataset_dir / label["frame_file"]

        image = cv2.imread(str(frame_path))
        if image is None:
            raise IOError(f"Failed to load: {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        angles = torch.tensor([label["theta_h"], label["theta_v"]], dtype=torch.float32)
        return image, angles


def load_labels(dataset_dir: Path) -> list[dict]:
    """Load labels.csv into a list of dicts."""
    labels_path = dataset_dir / "labels.csv"
    labels = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                "session": row["session"],
                "frame_file": row["frame_file"],
                "theta_h": float(row["theta_h"]),
                "theta_v": float(row["theta_v"]),
            })
    return labels


def split_by_session(labels: list[dict], val_ratio: float = 0.15) -> tuple[list, list]:
    """Split labels by session (not by frame) to prevent data leakage."""
    sessions = list(set(l["session"] for l in labels))
    random.seed(42)
    random.shuffle(sessions)

    n_val = max(1, int(len(sessions) * val_ratio))
    n_val = min(n_val, len(sessions) - 1)
    val_sessions = set(sessions[:n_val])
    train_sessions = set(sessions[n_val:])

    train_labels = [l for l in labels if l["session"] in train_sessions]
    val_labels = [l for l in labels if l["session"] in val_sessions]

    print(f"Train: {len(train_labels)} frames from {len(train_sessions)} sessions")
    print(f"Val:   {len(val_labels)} frames from {len(val_sessions)} sessions")

    return train_labels, val_labels


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train gaze model from Pi dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Prepared dataset directory")
    parser.add_argument("--distance", type=int, required=True, help="Distance in cm")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    dataset_dir = Path(args.dataset_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(dataset_dir)
    print(f"Total labels: {len(labels)}")

    train_labels, val_labels = split_by_session(labels, args.val_split)

    use_augment = not args.no_augment
    train_dataset = PiSmoothPursuitDataset(
        train_labels, dataset_dir,
        transform=PiSmoothPursuitDataset._default_transform(augment=use_augment),
    )
    val_dataset = PiSmoothPursuitDataset(
        val_labels, dataset_dir,
        transform=PiSmoothPursuitDataset._default_transform(augment=False),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = GazeMobileNet(pretrained=True, freeze_backbone=False).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from: {args.resume}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (epoch + 1) / args.warmup_epochs
        )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    model_filename = f"model_{args.distance}cm.pth"

    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Warmup: {args.warmup_epochs} epochs, early stop: {args.early_stop} epochs\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        if warmup_scheduler and epoch <= args.warmup_epochs:
            warmup_scheduler.step()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        if not warmup_scheduler or epoch > args.warmup_epochs:
            scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "distance_cm": args.distance,
            }, save_dir / model_filename)
            print(f"  Saved best model: {model_filename} (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs "
                  f"(best: epoch {best_epoch}, val={best_val_loss:.4f})")

        if args.early_stop > 0 and epochs_without_improvement >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"Model saved: {save_dir / model_filename}")


if __name__ == "__main__":
    main()

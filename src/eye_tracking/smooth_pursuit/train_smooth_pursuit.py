"""
Training script for smooth pursuit gaze model using collected data.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobilenet.model import GazeMobileNet


class SmoothPursuitDataset(Dataset):
    """Dataset from collected smooth pursuit recordings."""

    def __init__(self, session_dirs, transform=None):
        """
        Args:
            session_dirs: List of session directory paths
            transform: Image transforms
        """
        self.transform = transform or self._default_transform()
        self.samples = []

        for session_dir in session_dirs:
            session_path = Path(session_dir)
            metadata_file = session_path / "metadata.csv"

            if not metadata_file.exists():
                print(f"Warning: No metadata in {session_dir}")
                continue

            df = pd.read_csv(metadata_file)

            for _, row in df.iterrows():
                image_path = session_path / row["frame_id"]
                if image_path.exists():
                    self.samples.append(
                        {"image_path": str(image_path), "theta_h": row["theta_h"], "theta_v": row["theta_v"]}
                    )

        print(f"Loaded {len(self.samples)} samples from {len(session_dirs)} sessions")

    @staticmethod
    def _default_transform():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([sample["theta_h"], sample["theta_v"]], dtype=torch.float32)

        return image, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train smooth pursuit gaze model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing session folders")
    parser.add_argument("--distance", type=int, required=True, help="Distance in cm (e.g., 30)")
    parser.add_argument("--save_dir", type=str, default="mobilenet/checkpoints", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find all session directories
    data_path = Path(args.data_dir)
    session_dirs = [str(d) for d in data_path.glob("session_*") if (d / "metadata.csv").exists()]

    if len(session_dirs) == 0:
        print(f"No sessions found in {args.data_dir}")
        return

    print(f"Found {len(session_dirs)} sessions")

    # Split into train/val
    n_val = int(len(session_dirs) * args.val_split)
    val_dirs = session_dirs[:n_val]
    train_dirs = session_dirs[n_val:]

    # Create datasets
    train_dataset = SmoothPursuitDataset(train_dirs)
    val_dataset = SmoothPursuitDataset(val_dirs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = GazeMobileNet(pretrained=True, freeze_backbone=False)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    best_val_loss = float("inf")
    save_path = Path(__file__).parent.parent / args.save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            model_file = save_path / f"model_{args.distance}cm.pth"
            torch.save(checkpoint, model_file)
            print(f"Saved best model: {model_file}")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()


"""
Training script for smooth pursuit gaze model using collected data.
Uses SQLite database to track sessions and training runs.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobilenet.model import GazeMobileNet
from database import SmoothPursuitDB


class SmoothPursuitDataset(Dataset):
    """Dataset from collected smooth pursuit recordings using database."""

    def __init__(self, frame_records, base_data_dir, transform=None):
        """
        Args:
            frame_records: List of database frame records (from db.get_frames_for_sessions)
            base_data_dir: Base directory where session folders are stored
            transform: Image transforms
        """
        self.transform = transform or self._default_transform()
        self.samples = []
        self.base_data_dir = Path(base_data_dir)

        for frame in frame_records:
            # Construct image path from session_id and frame_filename
            session_id = frame["session_id"]
            frame_filename = frame["frame_filename"]
            
            # Session directory format: session_{session_id}_{distance}cm
            # We need to find the actual session directory
            session_dir = self.base_data_dir / f"session_{session_id}_{frame['distance_cm']}cm"
            image_path = session_dir / frame_filename
            
            if image_path.exists():
                self.samples.append(
                    {
                        "image_path": str(image_path),
                        "theta_h": frame["theta_h"],
                        "theta_v": frame["theta_v"],
                    }
                )
            else:
                print(f"Warning: Image not found: {image_path}")

        print(f"Loaded {len(self.samples)} samples from database")

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

    # Connect to database
    db = SmoothPursuitDB()
    
    # Get all sessions for the specified distance
    sessions = db.get_sessions_by_distance(args.distance)
    
    if len(sessions) == 0:
        print(f"No sessions found for distance {args.distance}cm in database")
        db.close()
        return

    print(f"Found {len(sessions)} sessions for distance {args.distance}cm")
    
    # Get session database IDs
    session_db_ids = [s["id"] for s in sessions]
    
    # Split into train/val by session (not by frame)
    n_val = int(len(session_db_ids) * args.val_split)
    val_session_ids = session_db_ids[:n_val]
    train_session_ids = session_db_ids[n_val:]
    
    print(f"Training sessions: {len(train_session_ids)}, Validation sessions: {len(val_session_ids)}")
    
    # Get frames for each split
    train_frames = db.get_frames_for_sessions(train_session_ids)
    val_frames = db.get_frames_for_sessions(val_session_ids)
    
    if len(train_frames) == 0:
        print(f"No training frames found for distance {args.distance}cm")
        db.close()
        return
    
    if len(val_frames) == 0:
        print(f"Warning: No validation frames found. Using all data for training.")
        val_frames = train_frames[-int(len(train_frames) * 0.1):]  # Use 10% for validation
        train_frames = train_frames[:-int(len(train_frames) * 0.1)]

    # Create datasets
    data_path = Path(args.data_dir)
    train_dataset = SmoothPursuitDataset(train_frames, data_path)
    val_dataset = SmoothPursuitDataset(val_frames, data_path)

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
    
    # Record model and training run in database
    model_name = f"model_{args.distance}cm.pth"
    model_db_id = db.create_model(
        model_name=model_name,
        distance_cm=args.distance,
        model_path=str(model_file),
        best_val_loss=best_val_loss,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        notes=f"Trained on {len(train_dataset)} samples, validated on {len(val_dataset)} samples",
    )
    
    # Link training sessions to model
    db.link_training_sessions(model_db_id, train_session_ids, val_session_ids)
    
    print(f"\nModel recorded in database:")
    print(f"  Model ID: {model_db_id}")
    print(f"  Training sessions: {len(train_session_ids)}")
    print(f"  Validation sessions: {len(val_session_ids)}")
    
    db.close()


if __name__ == "__main__":
    main()


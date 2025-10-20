"""
Training script for MobileNet gaze estimation model.

Trains the model to predict gaze direction (vertical and horizontal angles)
using MSE loss on the Columbia Gaze Dataset.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.eye_tracking.Dataset.dataloader import get_dataloaders  # noqa: E402
from src.eye_tracking.mobilenet.model import create_model  # noqa: E402


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str, epoch: int
) -> float:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        pbar.set_postfix({"val_loss": f"{total_loss / (pbar.n + 1):.4f}"})

    return total_loss / len(val_loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: str,
    num_epochs: int,
    save_dir: Path,
) -> None:
    """
    Main training loop.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train
        save_dir: Directory to save checkpoints
    """
    best_val_loss = float("inf")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Checkpoints will be saved to: {save_dir}\n")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        # Print epoch summary
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pth")

        print()

    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")


def main():
    """Main training function."""
    # Calculate default paths relative to script location
    # This allows the script to work from any directory
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "Dataset"
    default_save_dir = script_dir / "checkpoints"

    parser = argparse.ArgumentParser(description="Train MobileNet for gaze estimation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(default_save_dir),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone during training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        entity="3YP",
        project="gaze-estimation",
        config={
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "lr": args.lr,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
        },
    )

    # Set random seed
    torch.manual_seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create model
    print("Creating model...")
    model = create_model(
        pretrained=args.pretrained, freeze_backbone=args.freeze_backbone, device=device
    )

    # Loss function - MSE for regression
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()

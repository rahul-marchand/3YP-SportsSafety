"""Training script for RITnet on OpenEDS dataset."""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pupil_segmentation.Dataset.dataloader import get_dataloaders  # noqa: E402
from src.pupil_segmentation.evaluation.metrics import compute_dice, compute_iou  # noqa: E402
from src.pupil_segmentation.losses import get_loss  # noqa: E402
from src.pupil_segmentation.models import create_model  # noqa: E402


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (batch_idx + 1):.4f}"})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> tuple[float, float, float]:
    """Validate and return (loss, mean_iou, mean_dice)."""
    model.eval()
    total_loss = 0.0
    all_iou = []
    all_dice = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # Compute metrics
        preds = outputs.argmax(dim=1)
        for pred, mask in zip(preds, masks, strict=False):
            iou = compute_iou(pred, mask)
            dice = compute_dice(pred, mask)
            # Average over pupil and iris (skip background and sclera)
            all_iou.append((iou["pupil"] + iou["iris"]) / 2)
            all_dice.append((dice["pupil"] + dice["iris"]) / 2)

        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    mean_iou = sum(all_iou) / len(all_iou) if all_iou else 0.0
    mean_dice = sum(all_dice) / len(all_dice) if all_dice else 0.0

    return total_loss / len(val_loader), mean_iou, mean_dice


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
    use_wandb: bool = True,
) -> None:
    """Main training loop."""
    best_val_iou = 0.0
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV logging
    csv_path = save_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"]
    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    print(f"\nTraining for {num_epochs} epochs on {device}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)} samples")
    print(f"Checkpoints: {save_dir}\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device, epoch)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        # Log
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "lr": lr,
        }
        if use_wandb:
            wandb.log(metrics)
        csv_writer.writerow({"epoch": epoch, **metrics})
        csv_file.flush()

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {train_loss:.4f}/{val_loss:.4f}, "
            f"IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, LR: {lr:.6f}"
        )

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                },
                save_dir / "best_model.pth",
            )
            print(f"  -> Saved best model (IoU: {val_iou:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                save_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    csv_file.close()
    print(f"\nTraining complete. Best IoU: {best_val_iou:.4f}")
    print(f"Metrics saved to {csv_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_save_dir = script_dir / "checkpoints"

    parser = argparse.ArgumentParser(description="Train segmentation model for pupil/iris")
    parser.add_argument("--data_dir", type=Path, required=True, help="OpenEDS dataset directory")
    parser.add_argument("--save_dir", type=Path, default=default_save_dir)
    parser.add_argument(
        "--model", type=str, default="ritnet", help="Model architecture (ritnet, unet)"
    )
    parser.add_argument(
        "--loss", type=str, default="ce", help="Loss function (ce, dice, ce_dice, compound)"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--no_preprocessing", action="store_true", help="Disable gamma+CLAHE preprocessing"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            entity="3YP",
            project="pupil-segmentation",
            config=vars(args),
        )

    # Data
    apply_preprocessing = not args.no_preprocessing
    print(f"Loading dataset... (preprocessing={'on' if apply_preprocessing else 'off'})")
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_preprocessing=apply_preprocessing,
    )

    # Model
    print(f"Creating {args.model}...")
    model = create_model(args.model, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Loss
    print(f"Loss: {args.loss}")
    criterion = get_loss(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.num_epochs,
        args.save_dir,
        use_wandb,
    )


if __name__ == "__main__":
    main()

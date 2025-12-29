"""
Training script for smooth pursuit gaze model using collected data.
Uses SQLite database to track sessions and training runs.
"""

import argparse
import random
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
            # Construct image path from session_id_string and frame_filename
            # Use session_id_string (from sessions table) not session_id (database ID)
            # sqlite3.Row supports dict-style access - columns from JOIN query
            try:
                session_id_string = frame["session_id_string"]
            except (KeyError, IndexError):
                # Fallback to session_id if session_id_string not available
                session_id_string = frame["session_id"]
            
            frame_filename = frame["frame_filename"]
            
            try:
                distance_cm = frame["session_distance_cm"]
            except (KeyError, IndexError):
                # Fallback to distance_cm from frames table
                distance_cm = frame["distance_cm"]
            
            # Session directory format: session_{session_id_string}_{distance}cm
            session_dir = self.base_data_dir / f"session_{session_id_string}_{distance_cm}cm"
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
    def _default_transform(augment=False):
        """
        Default transform for training/validation.
        Note: Images are already 224x224 when saved (resized during capture),
        so no resize is needed here.
        
        Args:
            augment: If True, applies data augmentation (for training).
                    If False, only basic transforms (for validation).
                    Augmentation happens in-memory, does NOT modify original files.
        """
        if augment:
            # Training: Add data augmentation to improve generalization
            # All augmentation happens in-memory - original files are never modified
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),  # Small rotation for robustness
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random erasing for regularization
            ])
        else:
            # Validation: No augmentation, just normalize
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image with error handling
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image {sample['image_path']}: {e}")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([sample["theta_h"], sample["theta_v"]], dtype=torch.float32)

        return image, label


def train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
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
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
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
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for learning rate")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (model path or 'auto' for latest)")
    parser.add_argument("--fine_tune", type=str, default=None, help="Fine-tune from a saved model (model path)")

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
    
    # Split into train/val by session (not by frame) - use random shuffle to prevent data leakage
    # Shuffle sessions to avoid temporal bias if sessions are ordered chronologically
    random.seed(42)  # For reproducibility
    shuffled_ids = session_db_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Handle edge case: if only 1 session, we'll split frames instead (handled later)
    if len(shuffled_ids) == 1:
        print(f"Warning: Only 1 session found. Will split frames within this session for train/val.")
        train_session_ids = shuffled_ids  # Use the session for training
        val_session_ids = []  # Will split frames later
    else:
        n_val = max(1, int(len(shuffled_ids) * args.val_split))  # Ensure at least 1 validation session if possible
        # Make sure we don't put all sessions in validation
        n_val = min(n_val, len(shuffled_ids) - 1)  # Keep at least 1 session for training
        val_session_ids = shuffled_ids[:n_val]
        train_session_ids = shuffled_ids[n_val:]
    
    print(f"Training sessions: {len(train_session_ids)}, Validation sessions: {len(val_session_ids)}")
    
    # Get frames for each split
    train_frames = db.get_frames_for_sessions(train_session_ids) if train_session_ids else []
    val_frames = db.get_frames_for_sessions(val_session_ids) if val_session_ids else []
    
    # Handle single session case: split frames within the session
    if len(train_session_ids) == 1 and len(val_session_ids) == 0:
        print(f"Splitting frames within single session for train/val split.")
        random.seed(42)
        all_frames = train_frames.copy()
        random.shuffle(all_frames)
        n_val_frames = max(1, int(len(all_frames) * args.val_split))
        val_frames = all_frames[:n_val_frames]
        train_frames = all_frames[n_val_frames:]
    
    if len(train_frames) == 0:
        print(f"No training frames found for distance {args.distance}cm")
        db.close()
        return
    
    if len(val_frames) == 0:
        print(f"Warning: No validation frames found. Splitting training data randomly.")
        # Randomly split training frames if no validation sessions had frames
        random.seed(42)
        all_frames = train_frames.copy()
        random.shuffle(all_frames)
        n_val_frames = max(1, int(len(all_frames) * args.val_split))  # Use validation split ratio
        val_frames = all_frames[:n_val_frames]
        train_frames = all_frames[n_val_frames:]

    # Create datasets with appropriate transforms
    # Training: Use augmentation (in-memory, doesn't modify original files)
    # Validation: No augmentation
    data_path = Path(args.data_dir)
    train_dataset = SmoothPursuitDataset(
        train_frames, 
        data_path, 
        transform=SmoothPursuitDataset._default_transform(augment=not args.no_augment)
    )
    val_dataset = SmoothPursuitDataset(
        val_frames, 
        data_path, 
        transform=SmoothPursuitDataset._default_transform(augment=False)  # Never augment validation
    )

    # DataLoader with num_workers=4 for parallel loading (images loaded from disk efficiently)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = GazeMobileNet(pretrained=True, freeze_backbone=False)
    model = model.to(device)
    
    start_epoch = 1
    
    # Handle resume/fine-tune options
    if args.resume:
        # Resume training from checkpoint (continues from where it left off)
        if args.resume == "auto":
            # Find the latest model for this distance
            checkpoint_path = save_path / f"model_{args.distance}cm.pth"
        else:
            checkpoint_path = Path(args.resume)
        
        if checkpoint_path.exists():
            print(f"\n📂 RESUMING FROM CHECKPOINT: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(f"   Loaded model from epoch {checkpoint.get('epoch', '?')}")
            print(f"   Previous validation loss: {best_val_loss:.4f}")
            print(f"   Continuing from epoch {start_epoch}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print("   Starting fresh with ImageNet pretrained weights.")
    
    elif args.fine_tune:
        # Fine-tune from a saved model (starts epoch count from 1, but uses saved weights)
        fine_tune_path = Path(args.fine_tune)
        if not fine_tune_path.exists():
            # Try to find in checkpoints directory
            fine_tune_path = save_path / args.fine_tune
        
        if fine_tune_path.exists():
            print(f"\n🔧 FINE-TUNING FROM: {fine_tune_path}")
            checkpoint = torch.load(fine_tune_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            original_distance = None
            # Try to get the distance this model was trained on
            model_name = fine_tune_path.stem
            if "cm" in model_name:
                try:
                    original_distance = int(model_name.split("_")[-1].replace("cm", ""))
                except:
                    pass
            if original_distance and original_distance != args.distance:
                print(f"   ⚠ Note: Model was trained at {original_distance}cm, now training at {args.distance}cm")
            print(f"   Model weights loaded. Starting fresh training from epoch 1.")
        else:
            print(f"✗ Fine-tune model not found: {fine_tune_path}")
            db.close()
            return
    else:
        print("\n🆕 TRAINING FROM SCRATCH (ImageNet pretrained weights)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler with warmup
    # First use ReduceLROnPlateau for main training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Track warmup state
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        # Linear warmup: gradually increase LR from small value to args.lr over warmup_epochs
        # LambdaLR multiplies base LR (args.lr) by lambda value
        # epoch is 0-indexed in scheduler, so epoch 0 = first epoch
        # Lambda goes from (1/warmup_epochs) to 1.0, so LR goes from args.lr/warmup_epochs to args.lr
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) if epoch < args.warmup_epochs else 1.0
        )

    # Training loop
    best_val_loss = float("inf")
    save_path = Path(__file__).parent.parent / args.save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Model file path (will be updated when best model is saved)
    model_file = save_path / f"model_{args.distance}cm.pth"
    model_saved = False  # Track if model has been saved at least once

    # Early stopping tracking
    epochs_without_improvement = 0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if not args.no_augment:
        print("Data augmentation: ENABLED (in-memory, original files not modified)")
    else:
        print("Data augmentation: DISABLED")
    print(f"Learning rate warmup: {args.warmup_epochs} epochs")
    print(f"Early stopping patience: {args.early_stop_patience} epochs")
    print(f"Gradient clipping: max_norm={args.max_grad_norm}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Learning rate warmup
        if warmup_scheduler is not None and epoch <= args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Warmup LR: {current_lr:.6f}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=args.max_grad_norm)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate scheduler (after warmup)
        if epoch > args.warmup_epochs:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            model_file = save_path / f"model_{args.distance}cm.pth"
            
            # Save model with error handling
            try:
                torch.save(checkpoint, model_file)
                model_saved = True
                print(f"✓ Saved best model: {model_file} (val_loss improved to {val_loss:.4f})")
            except Exception as e:
                print(f"✗ ERROR: Failed to save model: {e}")
                print(f"  Attempted path: {model_file}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best: epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
        
        # Always save the first epoch's model (in case validation never improves)
        if epoch == 1 and not model_saved:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            try:
                torch.save(checkpoint, model_file)
                model_saved = True
                print(f"✓ Saved initial model: {model_file} (first epoch)")
            except Exception as e:
                print(f"✗ ERROR: Failed to save initial model: {e}")
        
        # Early stopping
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"\nEarly stopping triggered: No improvement for {args.early_stop_patience} epochs")
            print(f"Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
            break

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    # CRITICAL: Ensure model is saved before proceeding
    # This handles edge cases where model might not have been saved
    if not model_saved or not model_file.exists():
        print(f"\n⚠ WARNING: Model was not saved during training. Saving model now...")
        try:
            # If we have a best epoch, try to load that state (but we don't have it saved)
            # So we'll save the current model state
            checkpoint = {
                "epoch": best_epoch if best_epoch > 0 else 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss if best_val_loss != float("inf") else float("nan"),
                "train_loss": float("nan"),
            }
            torch.save(checkpoint, model_file)
            model_saved = True
            print(f"✓ Saved model: {model_file}")
        except Exception as e:
            print(f"✗ CRITICAL ERROR: Failed to save model: {e}")
            print(f"  Attempted path: {model_file.absolute()}")
            print(f"  Training completed but model could not be saved!")
            import traceback
            traceback.print_exc()
            db.close()
            return
    
    # Verify model file exists and is readable
    if not model_file.exists():
        print(f"✗ CRITICAL ERROR: Model file does not exist after save attempt: {model_file.absolute()}")
        print(f"  Training completed but model was not saved!")
        db.close()
        return
    
    # Verify model file is not empty
    file_size = model_file.stat().st_size
    if file_size == 0:
        print(f"✗ CRITICAL ERROR: Model file is empty: {model_file.absolute()}")
        print(f"  Model file exists but contains no data!")
        db.close()
        return
    
    # Verify model can be loaded (final check)
    try:
        test_checkpoint = torch.load(model_file, map_location="cpu")
        if "model_state_dict" not in test_checkpoint:
            print(f"✗ CRITICAL ERROR: Model file is corrupted - missing 'model_state_dict'")
            db.close()
            return
        print(f"✓ Model file verified and loadable: {model_file}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  Saved at epoch: {test_checkpoint.get('epoch', 'unknown')}")
        print(f"  Validation loss: {test_checkpoint.get('val_loss', 'unknown'):.4f}")
    except Exception as e:
        print(f"✗ CRITICAL ERROR: Model file cannot be loaded: {e}")
        print(f"  Model file may be corrupted!")
        db.close()
        return
    
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


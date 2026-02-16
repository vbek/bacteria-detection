"""
Main training script for Mask R-CNN
"""
import os
import json
import torch
from tqdm import tqdm

# Import from our modules
from config import (
    device, dataset_root, annotations_root, save_dir,
    num_classes, num_epochs, batch_size, learning_rate, weight_decay,
    print_config
)
from dataset import setup_datasets
from model import setup_model
from evaluate import evaluate


def train():
    """Main training function"""
    
    # Print configuration
    print_config()
    
    # Setup datasets
    train_loader, val_loader, test_loader, train_ann, val_ann, test_ann = setup_datasets(
        dataset_root, annotations_root, batch_size
    )
    
    # Setup model, optimizer, and load checkpoint
    model, optimizer, scheduler, start_epoch, checkpoint_path = setup_model(
        num_classes, device, learning_rate, weight_decay, save_dir
    )
    
    # Training loop
    train_log = []
    log_path = os.path.join(save_dir, "training_log.json")
    
    # Track best model
    best_mAP = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, num_epochs):
        # === Training ===
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)

        # === Validation ===
        val_mAP, val_IoU, val_loss = evaluate(model, val_loader, ann_file=val_ann, device=device)

        # === Update learning rate based on validation mAP ===
        scheduler.step(val_mAP)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_mAP:.4f} | "
            f"Val IoU: {val_IoU:.4f} | "
            f"LR: {current_lr:.2e}\n"
        )

        # === Save best model ===
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_mAP": val_mAP,
                "val_IoU": val_IoU,
                "val_loss": val_loss
            }
            torch.save(best_checkpoint, best_model_path)
            print(f"🌟 New best model! Val mAP: {val_mAP:.4f}")
            print(f"   → Saved to: {best_model_path}\n")

        # === Save checkpoint every 2 epochs ===
        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, epoch_checkpoint_path)
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint for epoch {epoch+1}")
            print(f"   → {epoch_checkpoint_path}")
            print(f"   → {checkpoint_path}\n")

        # === Log metrics ===
        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_mAP": val_mAP,
            "val_IoU": val_IoU
        })

        # === Save log after each epoch ===
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model checkpoints saved in: {save_dir}")
    print(f"Training log saved at: {log_path}")
    print(f"{'='*70}\n")

    # === Final Test Evaluation ===
    print(f"{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}\n")

    test_mAP, test_IoU, test_loss = evaluate(model, test_loader, ann_file=test_ann, device=device)

    print(
        f"\nTest Results:\n"
        f"  Test Loss: {test_loss:.4f}\n"
        f"  Test mAP: {test_mAP:.4f}\n"
        f"  Test IoU: {test_IoU:.4f}\n"
    )

    # Save test results
    test_results = {
        "test_loss": test_loss,
        "test_mAP": test_mAP,
        "test_IoU": test_IoU
    }

    test_results_path = os.path.join(save_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"Test results saved at: {test_results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train()

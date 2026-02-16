"""
Configuration file for Mask R-CNN training
"""
import os
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset paths (default configuration)
dataset_root = "./dataset"
annotations_root = "./dataset/annotations"
save_dir = "./model_latest"
os.makedirs(save_dir, exist_ok=True)

# Model parameters
num_classes = 3  # background + ecoli + salmonella (COCO format: 0=background, 1=ecoli, 2=salmonella)
num_epochs = 20
batch_size = 2
learning_rate = 1e-4
weight_decay = 1e-4

# Print configuration
def print_config():
    """Print training configuration"""
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset root: {dataset_root}")
    print(f"Annotations root: {annotations_root}")
    print(f"Save directory: {save_dir}")
    print(f"Number of classes: {num_classes} (background + ecoli + salmonella)")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*70}\n")

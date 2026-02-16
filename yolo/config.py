"""
Configuration file for YOLOv8 Segmentation training
"""
import os
import torch

# Device configuration
device = "0" if torch.cuda.is_available() else "cpu"  # YOLO uses string for device
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
dataset_root = "./dataset"
data_yaml_path = "./data.yaml"  # Will be created automatically

# Model parameters
model_name = "yolov8m-seg"  
num_classes = 2  # ecoli and salmonella (YOLO doesn't count background)

# Training parameters
num_epochs = 35
batch_size = 8
image_size = 1024
save_period = 5  # Save checkpoint every N epochs

# Project organization
project_dir = "./runs"
experiment_name = "yolo_segmentation"

# Augmentation settings
use_augment = True
use_mosaic = True

# Class names and IDs (YOLO format)
# YOLO uses 0-indexed classes (no background class)
class_names = {
    0: "ecoli",       # Class 0 in your .txt label files
    1: "salmonella"   # Class 1 in your .txt label files
}


def print_config():
    """Print training configuration"""
    print(f"\n{'='*70}")
    print("YOLO TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset root: {dataset_root}")
    print(f"Data YAML: {data_yaml_path}")
    print(f"Model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"  - Class 0: ecoli")
    print(f"  - Class 1: salmonella")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Save period: {save_period} epochs")
    print(f"Project dir: {project_dir}")
    print(f"Experiment name: {experiment_name}")
    print(f"{'='*70}\n")

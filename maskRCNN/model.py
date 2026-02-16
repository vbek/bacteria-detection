"""
Model setup utilities for Mask R-CNN
"""
import os
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def setup_model(num_classes, device, learning_rate, weight_decay, save_dir):
    """
    Setup Mask R-CNN model, optimizer, scheduler, and load checkpoint if available
    
    Args:
        num_classes: Number of classes (including background)
        device: Device to run on (cuda/cpu)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        save_dir: Directory to save/load checkpoints
        
    Returns:
        model, optimizer, scheduler, start_epoch, checkpoint_path
    """
    # Pretrained Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model.to(device)
    print(f"Model loaded and moved to {device}\n")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler (ReduceLROnPlateau with patience=3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # We want to maximize validation mAP
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        verbose=True,
        min_lr=1e-7
    )

    # Resume from checkpoint
    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔁 Resumed from epoch {start_epoch}")
        if "scheduler" in checkpoint:
            print(f"   Scheduler state restored")
        print()
    else:
        print("🆕 Starting fresh training.\n")

    return model, optimizer, scheduler, start_epoch, checkpoint_path

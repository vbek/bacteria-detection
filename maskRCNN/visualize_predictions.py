"""
Visualization script to display model predictions side-by-side with original images
Shows: Original image (left) | Predicted masks overlaid on image (right)
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import config
from config import device, num_classes


def load_model(checkpoint_path, device):
    """
    Load trained Mask R-CNN model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded model
    """
    # Create model architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # Replace predictors
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
    if "val_mAP" in checkpoint:
        print(f"   Val mAP: {checkpoint['val_mAP']:.4f}\n")
    
    return model


def load_image(image_path):
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        image_tensor: Tensor of shape (C, H, W)
        original_image: PIL Image (RGB)
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    return img_tensor, img


def visualize_predictions(image_path, model, device, score_threshold=0.5, save_path=None):
    """
    Visualize predictions side-by-side with original image
    
    Args:
        image_path: Path to image file
        model: Trained Mask R-CNN model
        device: Device to run inference on
        score_threshold: Confidence threshold for predictions
        save_path: Optional path to save the visualization
    """
    # Load image
    img_tensor, original_img = load_image(image_path)
    img_name = os.path.basename(image_path)
    
    # Get predictions
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
    
    # Filter by score threshold
    keep = prediction['scores'] > score_threshold
    pred_boxes = prediction['boxes'][keep].cpu().numpy()
    pred_masks = prediction['masks'][keep].cpu().numpy()
    pred_labels = prediction['labels'][keep].cpu().numpy()
    pred_scores = prediction['scores'][keep].cpu().numpy()
    
    # Count instances per class
    num_ecoli = (pred_labels == 1).sum()
    num_salmonella = (pred_labels == 2).sum()
    
    # Convert to numpy for visualization
    img_np = np.array(original_img)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left side: Original image
    ax1.imshow(img_np)
    ax1.set_title(f"{img_name}", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right side: Predictions
    ax2.imshow(img_np)
    
    # Define colors for each class
    colors = {
        1: [1, 0, 0],      # Red for E. coli
        2: [0, 0, 1]       # Blue for Salmonella
    }
    
    class_names = {
        1: "E. coli",
        2: "Salmonella"
    }
    
    # Overlay masks
    overlay = img_np.copy().astype(np.float32) / 255.0
    
    for mask, label, score, box in zip(pred_masks, pred_labels, pred_scores, pred_boxes):
        # Get binary mask
        binary_mask = mask[0] > 0.5
        
        # Get color for this class
        color = colors.get(label, [0, 1, 0])  # Default green if unknown
        
        # Create colored mask overlay (semi-transparent)
        for c in range(3):
            overlay[:, :, c] = np.where(
                binary_mask,
                overlay[:, :, c] * 0.5 + color[c] * 0.5,
                overlay[:, :, c]
            )
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Add label with confidence
        label_text = f"{class_names.get(label, 'Unknown')}: {score:.2f}"
        ax2.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    # Display overlay
    ax2.imshow(overlay)
    
    # Title with instance counts
    title = f"E. coli: {num_ecoli} | Salmonella: {num_salmonella}"
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add legend
    red_patch = mpatches.Patch(color='red', label='E. coli')
    blue_patch = mpatches.Patch(color='blue', label='Salmonella')
    ax2.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return num_ecoli, num_salmonella


def visualize_batch(image_paths, model, device, score_threshold=0.5, output_dir="visualizations"):
    """
    Visualize predictions for multiple images
    
    Args:
        image_paths: List of image paths
        model: Trained Mask R-CNN model
        device: Device to run inference on
        score_threshold: Confidence threshold for predictions
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VISUALIZING PREDICTIONS")
    print(f"{'='*70}\n")
    
    total_ecoli = 0
    total_salmonella = 0
    
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{img_name}_prediction.png")
        
        num_ecoli, num_salmonella = visualize_predictions(
            img_path, model, device, score_threshold, save_path
        )
        
        total_ecoli += num_ecoli
        total_salmonella += num_salmonella
        
        print(f"  {os.path.basename(img_path)}: E. coli={num_ecoli}, Salmonella={num_salmonella}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total E. coli detected: {total_ecoli}")
    print(f"Total Salmonella detected: {total_salmonella}")
    print(f"Visualizations saved in: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main visualization function"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Visualize Mask R-CNN predictions")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (e.g., best_model.pth)")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                        help="Path(s) to image file(s) or directory containing images")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations (default: visualizations)")
    parser.add_argument("--show", action="store_true",
                        help="Display images instead of saving")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, device)
    
    # Collect image paths
    image_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    for path in args.images:
        if os.path.isdir(path):
            # Directory - get all images
            for ext in valid_extensions:
                image_paths.extend(glob.glob(os.path.join(path, f"*{ext}")))
                image_paths.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        elif os.path.isfile(path):
            # Single file
            image_paths.append(path)
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images\n")
    
    # Visualize
    if args.show:
        # Show images one by one
        for img_path in image_paths:
            print(f"Processing: {os.path.basename(img_path)}")
            visualize_predictions(img_path, model, device, args.score_threshold, save_path=None)
    else:
        # Save all visualizations
        visualize_batch(image_paths, model, device, args.score_threshold, args.output_dir)


if __name__ == "__main__":
    main()

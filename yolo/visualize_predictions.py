"""
Visualization script for YOLOv8 predictions
Shows: Original image (left) | Predicted masks overlaid on image (right)
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
from PIL import Image


def visualize_prediction(image_path, model, save_path=None, conf_threshold=0.5):
    """
    Visualize YOLO predictions side-by-side with original image
    
    Args:
        image_path: Path to image file
        model: Trained YOLO model
        save_path: Optional path to save visualization
        conf_threshold: Confidence threshold for predictions
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_name = os.path.basename(image_path)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )[0]
    
    # Get predictions
    boxes = results.boxes
    masks = results.masks
    
    # Count instances per class
    num_ecoli = 0
    num_salmonella = 0
    
    if boxes is not None and len(boxes) > 0:
        for cls in boxes.cls:
            if int(cls) == 0:
                num_ecoli += 1
            elif int(cls) == 1:
                num_salmonella += 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original image
    ax1.imshow(img_np)
    ax1.set_title(f"{img_name}", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Predictions
    overlay = img_np.copy().astype(np.float32) / 255.0
    
    # Define colors
    colors = {
        0: [1, 0, 0],  # Red for E. coli
        1: [0, 0, 1]   # Blue for Salmonella
    }
    
    class_names = {
        0: "E. coli",
        1: "Salmonella"
    }
    
    # Draw masks and boxes
    if masks is not None and len(masks) > 0:
        for i, (mask, box) in enumerate(zip(masks.data, boxes)):
            # Get class and confidence
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            # Get color
            color = colors.get(cls, [0, 1, 0])
            
            # Draw mask
            mask_np = mask.cpu().numpy()
            
            # Resize mask to image size if needed
            if mask_np.shape != img_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
            
            # Apply colored overlay
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_np > 0.5,
                    overlay[:, :, c] * 0.5 + color[c] * 0.5,
                    overlay[:, :, c]
                )
            
            # Draw bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax2.add_patch(rect)
            
            # Add label
            label_text = f"{class_names.get(cls, 'Unknown')}: {conf:.2f}"
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
    
    # Title with counts
    title = f"E. coli: {num_ecoli} | Salmonella: {num_salmonella}"
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Legend
    red_patch = mpatches.Patch(color='red', label='E. coli')
    blue_patch = mpatches.Patch(color='blue', label='Salmonella')
    ax2.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return num_ecoli, num_salmonella


def visualize_batch(image_paths, model, output_dir="visualizations", conf_threshold=0.5):
    """
    Visualize predictions for multiple images
    
    Args:
        image_paths: List of image paths
        model: Trained YOLO model
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold
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
        
        num_ecoli, num_salmonella = visualize_prediction(
            img_path, model, save_path, conf_threshold
        )
        
        total_ecoli += num_ecoli
        total_salmonella += num_salmonella
        
        print(f" {os.path.basename(img_path)}: E. coli={num_ecoli}, Salmonella={num_salmonella}")
    
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
    from config import project_dir, experiment_name
    
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (default: best.pt from training)")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                        help="Path(s) to image file(s) or directory")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory (default: visualizations)")
    parser.add_argument("--show", action="store_true",
                        help="Show images instead of saving")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(project_dir, experiment_name, "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(project_dir, experiment_name, "weights", "last.pt")
        if not os.path.exists(model_path):
            print("No trained model found. Please specify --model")
            return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Collect image paths
    image_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    for path in args.images:
        if os.path.isdir(path):
            for ext in valid_extensions:
                image_paths.extend(glob.glob(os.path.join(path, f"*{ext}")))
                image_paths.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        elif os.path.isfile(path):
            image_paths.append(path)
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images\n")
    
    # Visualize
    if args.show:
        for img_path in image_paths:
            print(f"Processing: {os.path.basename(img_path)}")
            visualize_prediction(img_path, model, None, args.conf)
    else:
        visualize_batch(image_paths, model, args.output_dir, args.conf)


if __name__ == "__main__":
    main()

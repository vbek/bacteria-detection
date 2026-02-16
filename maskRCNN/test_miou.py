"""
Evaluation script to compute mIoU (mean Intersection over Union) on test images
Supports PNG, JPG, and TIF image formats
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

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
        print(f"   Val mAP: {checkpoint['val_mAP']:.4f}")
    
    return model


def load_image(image_path):
    """
    Load image from file (supports PNG, JPG, TIF)
    
    Args:
        image_path: Path to image file
        
    Returns:
        image_tensor: Tensor of shape (C, H, W)
        original_image: PIL Image
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    
    return img_tensor, img


def get_test_images(test_dir, annotation_file=None):
    """
    Get all test images from directory
    
    Args:
        test_dir: Directory containing test images
        annotation_file: Optional COCO annotation file for ground truth
        
    Returns:
        image_paths: List of image paths
        annotations_dict: Dictionary mapping image_id to annotations (if annotation_file provided)
    """
    # Supported extensions
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    image_paths = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} test images")
    
    # Load annotations if provided
    annotations_dict = {}
    if annotation_file and os.path.exists(annotation_file):
        coco = COCO(annotation_file)
        print(f"Loaded annotations from: {annotation_file}")
        
        # Create mapping from filename to annotations
        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            annotations_dict[img_info['file_name']] = {
                'image_id': img_id,
                'annotations': anns,
                'height': img_info['height'],
                'width': img_info['width']
            }
    
    return image_paths, annotations_dict


def compute_iou(pred_mask, gt_mask):
    """
    Compute IoU between predicted and ground truth mask
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
        
    Returns:
        iou: Intersection over Union score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_miou(model, image_paths, annotations_dict, device, score_threshold=0.5, iou_threshold=0.5):
    """
    Evaluate mean IoU on test images
    
    Args:
        model: Trained Mask R-CNN model
        image_paths: List of image paths
        annotations_dict: Dictionary with ground truth annotations
        device: Device to run inference on
        score_threshold: Confidence threshold for predictions
        iou_threshold: Threshold for mask binarization
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    all_ious = []
    per_class_ious = {1: [], 2: []}  # 1: ecoli, 2: salmonella
    matched_predictions = 0
    total_gt_instances = 0
    total_pred_instances = 0
    
    print(f"\n{'='*70}")
    print("EVALUATING mIoU ON TEST IMAGES")
    print(f"{'='*70}\n")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load image
        img_tensor, _ = load_image(img_path)
        img_name = os.path.basename(img_path)
        
        # Get predictions
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        
        # Filter by score threshold
        keep = prediction['scores'] > score_threshold
        pred_boxes = prediction['boxes'][keep].cpu().numpy()
        pred_masks = prediction['masks'][keep].cpu().numpy()
        pred_labels = prediction['labels'][keep].cpu().numpy()
        pred_scores = prediction['scores'][keep].cpu().numpy()
        
        total_pred_instances += len(pred_masks)
        
        # Get ground truth if available
        if img_name in annotations_dict:
            gt_data = annotations_dict[img_name]
            gt_anns = gt_data['annotations']
            total_gt_instances += len(gt_anns)
            
            # Process each ground truth instance
            for gt_ann in gt_anns:
                if 'segmentation' not in gt_ann:
                    continue
                
                # Decode ground truth mask - handle both polygon and RLE formats
                seg = gt_ann['segmentation']
                gt_label = gt_ann['category_id']
                
                # Handle different segmentation formats
                if isinstance(seg, dict):
                    # RLE format (already encoded)
                    if 'counts' in seg and 'size' in seg:
                        gt_mask = mask_utils.decode(seg)
                    else:
                        # Invalid format, skip this annotation
                        continue
                elif isinstance(seg, list):
                    # Polygon format - need to convert to mask
                    if len(seg) > 0:
                        # Convert polygon to RLE then to mask
                        h, w = gt_data['height'], gt_data['width']
                        rles = mask_utils.frPyObjects(seg, h, w)
                        rle = mask_utils.merge(rles)
                        gt_mask = mask_utils.decode(rle)
                    else:
                        # Empty polygon, skip
                        continue
                else:
                    # Unknown format, skip
                    continue
                
                # Find best matching prediction
                best_iou = 0.0
                best_pred_idx = -1
                
                for pred_idx, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
                    # Only match same class
                    if pred_label != gt_label:
                        continue
                    
                    # Binarize predicted mask
                    pred_mask_binary = (pred_mask[0] > iou_threshold).astype(np.uint8)
                    
                    # Compute IoU
                    iou = compute_iou(pred_mask_binary, gt_mask)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                # Record IoU if match found
                if best_pred_idx >= 0:
                    all_ious.append(best_iou)
                    per_class_ious[gt_label].append(best_iou)
                    matched_predictions += 1
    
    # Compute metrics
    if len(all_ious) > 0:
        mean_iou = np.mean(all_ious)
        median_iou = np.median(all_ious)
    else:
        mean_iou = 0.0
        median_iou = 0.0
    
    # Per-class metrics
    class_metrics = {}
    class_names = {1: "ecoli", 2: "salmonella"}
    for class_id, class_name in class_names.items():
        if len(per_class_ious[class_id]) > 0:
            class_metrics[class_name] = {
                "mean_iou": float(np.mean(per_class_ious[class_id])),
                "median_iou": float(np.median(per_class_ious[class_id])),
                "num_instances": len(per_class_ious[class_id])
            }
        else:
            class_metrics[class_name] = {
                "mean_iou": 0.0,
                "median_iou": 0.0,
                "num_instances": 0
            }
    
    results = {
        "mean_iou": float(mean_iou),
        "median_iou": float(median_iou),
        "total_matched_instances": matched_predictions,
        "total_gt_instances": total_gt_instances,
        "total_pred_instances": total_pred_instances,
        "matching_rate": float(matched_predictions / max(1, total_gt_instances)),
        "per_class": class_metrics
    }
    
    return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate mIoU on test images")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to model checkpoint (e.g., best_model.pth)")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to COCO annotations file (optional)")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions (default: 0.5)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Threshold for mask binarization (default: 0.5)")
    parser.add_argument("--output", type=str, default="miou_results.json",
                        help="Output file for results (default: miou_results.json)")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, device)
    
    # Get test images
    image_paths, annotations_dict = get_test_images(args.test_dir, args.annotations)
    
    if len(image_paths) == 0:
        print("❌ No test images found!")
        return
    
    # Evaluate
    results = evaluate_miou(
        model, 
        image_paths, 
        annotations_dict, 
        device,
        args.score_threshold,
        args.iou_threshold
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Median IoU: {results['median_iou']:.4f}")
    print(f"Matched instances: {results['total_matched_instances']} / {results['total_gt_instances']}")
    print(f"Matching rate: {results['matching_rate']:.4f}")
    print(f"Total predictions: {results['total_pred_instances']}")
    
    print(f"\nPer-class metrics:")
    for class_name, metrics in results['per_class'].items():
        print(f"  {class_name.capitalize()}:")
        print(f"    Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"    Median IoU: {metrics['median_iou']:.4f}")
        print(f"    Instances: {metrics['num_instances']}")
    
    print(f"{'='*70}\n")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()

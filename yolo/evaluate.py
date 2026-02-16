"""
Evaluation script for YOLOv8 Segmentation model
"""
import os
from ultralytics import YOLO
from config import data_yaml_path, project_dir, experiment_name


def evaluate_model(model_path, split='test', save_json=True):
    """
    Evaluate YOLO model on specified split
    
    Args:
        model_path: Path to model weights
        split: Dataset split to evaluate on ('train', 'val', or 'test')
        save_json: Whether to save results as JSON
        
    Returns:
        results: Validation results object
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL ON {split.upper()} SET")
    print(f"{'='*70}\n")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Verify data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found!")
        return None
    
    # Run validation
    results = model.val(
        data=data_yaml_path,
        split=split,
        save_json=save_json,
        save_hybrid=True,
        plots=True,
        verbose=True,
        workers=1  # Reduce workers to avoid warning
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{split.upper()} SET RESULTS")
    print(f"{'='*70}")
    print(f"Box Metrics:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    if hasattr(results, 'seg'):
        print(f"\nSegmentation Metrics:")
        print(f"  mAP50: {results.seg.map50:.4f}")
        print(f"  mAP50-95: {results.seg.map:.4f}")
        print(f"  Precision: {results.seg.mp:.4f}")
        print(f"  Recall: {results.seg.mr:.4f}")
    
    print(f"\nPer-class mAP50 (Box):")
    if hasattr(results.box, 'ap_class_index'):
        for idx, class_idx in enumerate(results.box.ap_class_index):
            class_map = results.box.ap50[idx] if idx < len(results.box.ap50) else 0.0
            print(f"  Class {class_idx}: {class_map:.4f}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Segmentation model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (default: best.pt from training)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--no-json", action="store_true",
                        help="Don't save results as JSON")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Use best.pt from training directory
        model_path = os.path.join(project_dir, experiment_name, "weights", "best.pt")
        
        if not os.path.exists(model_path):
            # Try last.pt
            model_path = os.path.join(project_dir, experiment_name, "weights", "last.pt")
            
        if not os.path.exists(model_path):
            print(f"No trained model found in {os.path.join(project_dir, experiment_name)}")
            print("Please specify model path with --model or train a model first")
            return
    
    print(f"Model: {model_path}")
    
    # Evaluate
    evaluate_model(model_path, args.split, save_json=not args.no_json)


if __name__ == "__main__":
    main()

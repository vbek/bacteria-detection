"""
Training script for YOLOv8 Segmentation
Supports resuming from previous checkpoints
"""
import os
from ultralytics import YOLO
from config import (
    model_name, data_yaml_path, num_epochs, batch_size, image_size,
    device, project_dir, experiment_name, save_period,
    use_augment, use_mosaic, print_config
)


def get_latest_weights():
    """
    Find the latest weights to resume from
    
    Returns:
        Path to weights file or None
    """
    # Check for last.pt in the experiment directory
    last_weights = os.path.join(project_dir, experiment_name, "weights", "last.pt")
    
    if os.path.exists(last_weights):
        return last_weights
    
    return None


def train_yolo_segmentation(resume=True):
    """
    Train YOLOv8 segmentation model
    
    Args:
        resume: If True, resume from last.pt if available
    """
    print_config()
    
    print(f"{'='*70}")
    print("STARTING YOLO TRAINING")
    print(f"{'='*70}\n")
    
    # Check if we should resume
    latest_weights = get_latest_weights() if resume else None
    
    if latest_weights:
        print(f"Resuming training from: {latest_weights}\n")
        model = YOLO(latest_weights)
        resume_training = True
    else:
        print(f"Starting fresh training with: {model_name}\n")
        model = YOLO(f"{model_name}.pt")
        resume_training = False
    
    # Verify data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found!")
        print("Please run: python create_yaml.py")
        return
    
    # Train the model
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=num_epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=project_dir,
            name=experiment_name,
            exist_ok=True,           # Allow overwriting existing project
            pretrained=True,
            augment=use_augment,
            mosaic=use_mosaic,
            verbose=True,
            save=True,
            save_period=save_period,
            resume=resume_training,  # Resume from checkpoint
            plots=True,              # Save training plots
            val=True,                # Validate during training
            workers=0,
        )
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Results saved in: {os.path.join(project_dir, experiment_name)}")
        print(f"Best weights: {os.path.join(project_dir, experiment_name, 'weights', 'best.pt')}")
        print(f"Last weights: {os.path.join(project_dir, experiment_name, 'weights', 'last.pt')}")
        print(f"{'='*70}\n")
        
        # Validate on test set
        print(f"{'='*70}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*70}\n")
        
        best_model = YOLO(os.path.join(project_dir, experiment_name, "weights", "best.pt"))
        test_results = best_model.val(
            data=data_yaml_path,
            split='test',
            save_json=True,
            save_hybrid=True
        )
        
        print(f"\n{'='*70}")
        print("TEST SET RESULTS")
        print(f"{'='*70}")
        print(f"mAP50: {test_results.box.map50:.4f}")
        print(f"mAP50-95: {test_results.box.map:.4f}")
        if hasattr(test_results, 'seg'):
            print(f"Seg mAP50: {test_results.seg.map50:.4f}")
            print(f"Seg mAP50-95: {test_results.seg.map:.4f}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh training (don't resume from last.pt)")
    
    args = parser.parse_args()
    
    train_yolo_segmentation(resume=not args.no_resume)


if __name__ == "__main__":
    main()

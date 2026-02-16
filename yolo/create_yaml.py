"""
Script to create YOLO data.yaml file from dataset structure
"""
import os
import yaml
from config import dataset_root, data_yaml_path, num_classes, class_names


def create_data_yaml(dataset_root, output_path):
    """
    Create YOLO data.yaml file
    
    Args:
        dataset_root: Root directory of the dataset
        output_path: Path where data.yaml will be saved
    """
    # Convert to absolute paths
    dataset_root = os.path.abspath(dataset_root)
    
    # Define paths
    train_images = os.path.join(dataset_root, "train", "images")
    train_labels = os.path.join(dataset_root, "train", "labels")
    val_images = os.path.join(dataset_root, "val", "images")
    val_labels = os.path.join(dataset_root, "val", "labels")
    test_images = os.path.join(dataset_root, "test", "images")
    test_labels = os.path.join(dataset_root, "test", "labels")
    
    # Verify paths exist
    print("Verifying dataset paths...")
    for path, name in [(train_images, "Train images"), 
                       (train_labels, "Train labels"),
                       (val_images, "Val images"),
                       (val_labels, "Val labels"),
                       (test_images, "Test images"),
                       (test_labels, "Test labels")]:
        if os.path.exists(path):
            if 'images' in path:
                num_files = len([f for f in os.listdir(path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
                print(f"{name}: {path} ({num_files} files)")
            else:
                num_files = len([f for f in os.listdir(path) if f.endswith('.txt')])
                print(f"{name}: {path} ({num_files} files)")
        else:
            print(f"{name} NOT FOUND: {path}")
            if 'labels' in path:
                print(f"     WARNING: Labels directory missing. Make sure YOLO .txt files are in this location!")
            else:
                raise FileNotFoundError(f"Required path not found: {path}")
    
    # Create YAML content
    data = {
        'path': dataset_root,  # Root directory
        'train': 'train/images',  # Relative to path
        'val': 'val/images',      # Relative to path
        'test': 'test/images',    # Relative to path
        'nc': num_classes,        # Number of classes
        'names': [class_names[i] for i in range(num_classes)]  # Class names
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated data.yaml at: {output_path}")
    print("\nYAML content:")
    print("-" * 50)
    with open(output_path, 'r') as f:
        print(f.read())
    print("-" * 50)
    
    return output_path


def main():
    """Main function to create YAML"""
    print(f"\n{'='*70}")
    print("CREATING YOLO DATA.YAML FILE")
    print(f"{'='*70}\n")
    
    yaml_path = create_data_yaml(dataset_root, data_yaml_path)
    
    print(f"\n{'='*70}")
    print("DATASET READY!")
    print(f"{'='*70}")
    print("Your YOLO labels are already in place.")
    print("\nExpected structure:")
    print("  dataset/")
    print("    ├── train/")
    print("    │   ├── images/  (your images)")
    print("    │   └── labels/  (your .txt files)")
    print("    ├── val/")
    print("    │   ├── images/")
    print("    │   └── labels/")
    print("    └── test/")
    print("        ├── images/")
    print("        └── labels/")
    print(f"\n{'='*70}")
    print("NEXT STEP")
    print(f"{'='*70}")
    print("Start training:")
    print("  python train.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

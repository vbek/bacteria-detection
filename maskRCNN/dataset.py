"""
Dataset utilities for Mask R-CNN training
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from pycocotools import mask as mask_utils


def collate_fn(batch):
    """Custom collate function for batching"""
    return tuple(zip(*batch))


class CocoDatasetWrapper(CocoDetection):
    """Wrapper for COCO dataset to return tensors in the required format"""
    
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = F.to_tensor(img)
        
        # Get the actual COCO image_id from the dataset
        coco_img_id = self.ids[idx]
        
        # Get image dimensions
        img_height, img_width = img.shape[1], img.shape[2]

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

            if "segmentation" in ann:
                seg = ann["segmentation"]
                
                # Handle different segmentation formats
                if isinstance(seg, dict):
                    # RLE format (already encoded)
                    if 'counts' in seg and 'size' in seg:
                        mask = mask_utils.decode(seg)
                    else:
                        # Invalid format, create empty mask
                        mask = np.zeros((img_height, img_width), dtype=np.uint8)
                elif isinstance(seg, list):
                    # Polygon format - need to convert to mask
                    if len(seg) > 0:
                        # Convert polygon to RLE then to mask
                        rles = mask_utils.frPyObjects(seg, img_height, img_width)
                        rle = mask_utils.merge(rles)
                        mask = mask_utils.decode(rle)
                    else:
                        mask = np.zeros((img_height, img_width), dtype=np.uint8)
                else:
                    # Unknown format, create empty mask
                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([coco_img_id])  # Changed from idx to coco_img_id
        }

        return img, target


def setup_datasets(dataset_root, annotations_root, batch_size):
    """
    Setup datasets and data loaders
    
    Args:
        dataset_root: Root directory of the dataset
        annotations_root: Root directory of annotations
        batch_size: Batch size for training
        
    Returns:
        train_loader, val_loader, test_loader, train_ann, val_ann, test_ann
    """
    # Dataset paths
    train_img_dir = os.path.join(dataset_root, "train", "images")
    val_img_dir = os.path.join(dataset_root, "val", "images")
    test_img_dir = os.path.join(dataset_root, "test", "images")

    train_ann = os.path.join(annotations_root, "train.json")
    val_ann = os.path.join(annotations_root, "val.json")
    test_ann = os.path.join(annotations_root, "test.json")

    # Verify paths exist
    print("Verifying dataset paths...")
    for path, name in [(train_img_dir, "Train images"), (val_img_dir, "Val images"), 
                       (test_img_dir, "Test images"), (train_ann, "Train annotations"),
                       (val_ann, "Val annotations"), (test_ann, "Test annotations")]:
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name} NOT FOUND: {path}")
            raise FileNotFoundError(f"Required path not found: {path}")

    # Create datasets
    train_ds = CocoDatasetWrapper(train_img_dir, train_ann)
    val_ds = CocoDatasetWrapper(val_img_dir, val_ann)
    test_ds = CocoDatasetWrapper(test_img_dir, test_ann)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"\n{'='*70}")
    print(f"Dataset loaded successfully!")
    print(f"  Train images: {len(train_ds)}")
    print(f"  Val images: {len(val_ds)}")
    print(f"  Test images: {len(test_ds)}")
    print(f"{'='*70}\n")

    return train_loader, val_loader, test_loader, train_ann, val_ann, test_ann

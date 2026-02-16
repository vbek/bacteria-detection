"""
Evaluation utilities for Mask R-CNN
"""
import torch
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(model, data_loader, ann_file, device="cuda"):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Mask R-CNN model
        data_loader: DataLoader for validation/test set
        ann_file: Path to annotation file
        device: Device to run on (cuda/cpu)
        
    Returns:
        mAP: Mean Average Precision at IoU=[0.5:0.95]
        mean_IoU: Mean IoU at IoU=0.5
        val_loss: Average validation loss
    """
    cocoGt = COCO(ann_file)
    results = []
    total_loss = 0.0
    num_batches = 0

    model.eval()

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        # Compute validation loss
        with torch.no_grad():
            model.train()  # temporarily switch to train mode for loss
            loss_dict = model(images, targets)
            model.eval()   # switch back to eval
            val_loss_batch = sum(loss for loss in loss_dict.values())
            total_loss += val_loss_batch.item()
            num_batches += 1

        # Get predictions for COCO metrics
        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = target["image_id"].item()
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                mask_bin = (mask[0] > 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_bin))
                rle["counts"] = rle["counts"].decode("utf-8")

                x_min, y_min, x_max, y_max = box.tolist()
                w, h = x_max - x_min, y_max - y_min

                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x_min, y_min, w, h],
                    "score": float(score),
                    "segmentation": rle
                })

    # Compute average validation loss
    val_loss = total_loss / max(1, num_batches)

    # COCO Evaluation
    if len(results) == 0:
        print("No predictions generated!")
        return 0.0, 0.0, val_loss

    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP = cocoEval.stats[0]       # mAP@[0.5:0.95]
    mean_IoU = cocoEval.stats[1]  # mAP@0.5

    return mAP, mean_IoU, val_loss

# Mask R-CNN Training for E. coli and Salmonella Detection

This repository contains a PyTorch implementation of Mask R-CNN for instance segmentation of E. coli and Salmonella bacteria.

## Project Structure

```
.
├── config.py              # Configuration settings (paths, hyperparameters)
├── dataset.py             # Dataset loader and preprocessing
├── model.py               # Model setup and checkpoint handling
├── evaluate.py            # Evaluation metrics and validation
├── train.py               # Main training script
├── test_miou.py           # mIoU evaluation on test images
├── visualize_predictions.py  # Visualize predictions side-by-side
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vbek/bacteria-detection.git
cd <bacteria-detection>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Your dataset should follow this structure:
```
dataset/
├── train/
│   └── images/
├── val/
│   └── images/
├── test/
│   └── images/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

The annotation files should be in COCO format.

## Configuration

Edit `config.py` to customize:
- Dataset paths (`dataset_root`, `annotations_root`)
- Model save directory (`save_dir`)
- Training hyperparameters (`num_epochs`, `batch_size`, `learning_rate`, etc.)

### Default Configuration
- **Classes**: 3 (background + E. coli + Salmonella)
- **Epochs**: 20
- **Batch Size**: 2
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **LR Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
  - Automatically reduces learning rate by 50% if validation mAP doesn't improve for 3 epochs
  - Helps model converge better and avoid plateaus

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
1. Verify dataset paths
2. Load datasets and create data loaders
3. Initialize or resume model training (including scheduler state)
4. Automatically adjust learning rate when validation mAP plateaus
5. Save checkpoints every 2 epochs
6. Evaluate on validation set after each epoch
7. Perform final test evaluation after training

### Output Files

Training produces:
- `model_latest/best_model.pth` - **Best model** based on validation mAP
- `model_latest/checkpoint_epoch_*.pth` - Model checkpoints every 2 epochs
- `model_latest/latest_checkpoint.pth` - Most recent checkpoint for resuming
- `model_latest/training_log.json` - Training metrics per epoch
- `model_latest/test_results.json` - Final test set results

### Resume Training

Training **automatically resumes** from the latest checkpoint if available. The script will:
- Load the model state, optimizer state, scheduler state, and epoch number
- Continue training from where it left off
- Preserve all training history and learning rate schedule

Just run:
```bash
python train.py
```

The training will pick up from the last saved epoch.

### Model Checkpoints

The training script saves:
1. **Best Model** (`best_model.pth`): Saved whenever validation mAP improves
2. **Latest Checkpoint** (`latest_checkpoint.pth`): Updated every 2 epochs for resuming
3. **Epoch Checkpoints** (`checkpoint_epoch_*.pth`): Saved every 2 epochs

## Evaluating mIoU on Test Images

Use `test_miou.py` to evaluate mean Intersection over Union on test images. This script:
- Supports PNG, JPG, and TIF image formats
- Works with or without COCO annotations
- Computes per-class and overall mIoU metrics
- Matches predictions to ground truth instances

### Basic Usage

```bash
# With annotations (computes mIoU against ground truth)
python test_miou.py \
    --model model_latest/best_model.pth \
    --test_dir dataset/test/images \
    --annotations dataset/annotations/test.json \
    --output miou_results.json

# Without annotations (just runs inference)
python test_miou.py \
    --model model_latest/best_model.pth \
    --test_dir dataset/test/images \
    --output miou_results.json
```

### Advanced Options

```bash
python test_miou.py \
    --model model_latest/best_model.pth \
    --test_dir dataset/test/images \
    --annotations dataset/annotations/test.json \
    --score_threshold 0.7 \      # Confidence threshold (default: 0.5)
    --iou_threshold 0.5 \         # Mask binarization threshold (default: 0.5)
    --output miou_results.json
```

### Output Metrics

The evaluation provides:
- **Mean IoU**: Average IoU across all matched instances
- **Median IoU**: Median IoU for robustness to outliers
- **Per-class metrics**: Separate IoU for E. coli and Salmonella
- **Matching rate**: Percentage of ground truth instances matched
- **Instance counts**: Total predictions vs ground truth

## Visualizing Predictions

Use `visualize_predictions.py` to create side-by-side visualizations showing:
- **Left**: Original image with filename
- **Right**: Predicted masks overlaid on image with instance counts

The visualizations display:
- **Red masks**: E. coli instances
- **Blue masks**: Salmonella instances
- **Bounding boxes**: With confidence scores
- **Title**: Count of detected instances for each class

### Usage Examples

```bash
# Visualize a single image
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images dataset/test/images/image1.jpg

# Visualize multiple images
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images image1.jpg image2.jpg image3.jpg

# Visualize all images in a directory
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images dataset/test/images/

# Custom output directory
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images dataset/test/images/ \
    --output_dir my_visualizations

# Adjust confidence threshold
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images dataset/test/images/ \
    --score_threshold 0.7

# Show images interactively instead of saving
python visualize_predictions.py \
    --model model_latest/best_model.pth \
    --images image1.jpg \
    --show
```

### Output

Visualizations are saved as PNG files in the output directory (default: `visualizations/`):
```
visualizations/
├── image1_prediction.png
├── image2_prediction.png
└── image3_prediction.png
```

Each visualization shows:
- Original image name on the left
- Detected instances with masks and boxes on the right
- Instance counts: "E. coli: X | Salmonella: Y"

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pretrained**: COCO dataset weights
- **Detection Head**: Faster R-CNN box predictor
- **Segmentation Head**: Mask R-CNN predictor

## Learning Rate Scheduler

The training uses **ReduceLROnPlateau** scheduler with the following settings:
- **Metric**: Validation mAP (maximize)
- **Patience**: 3 epochs (waits 3 epochs before reducing LR)
- **Factor**: 0.5 (reduces LR by half when triggered)
- **Min LR**: 1e-7

**How it works:**
- Monitors validation mAP after each epoch
- If mAP doesn't improve for 3 consecutive epochs, learning rate is reduced by 50%
- Helps the model escape local minima and converge better
- Current learning rate is displayed in the training output

## Metrics

The evaluation includes:
- **Training Loss**: Average loss per epoch
- **Validation Loss**: Loss on validation set
- **mAP@[0.5:0.95]**: Mean Average Precision at IoU thresholds 0.5 to 0.95
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5

## Notes

- The code supports both relative paths (e.g., `../dataset`) and absolute paths
- Checkpoints are saved every 2 epochs to save disk space
- The model uses COCO-pretrained weights for transfer learning
- All code cells from the original notebook are preserved

## Citation

If you use this code, please cite the relevant papers for Mask R-CNN and PyTorch.

## License

[Add your license here]

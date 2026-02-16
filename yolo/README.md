# YOLOv8 Segmentation Training for E. coli and Salmonella Detection

Complete YOLOv8 instance segmentation training pipeline for bacteria detection.

## Project Structure

```
yolo/
├── config.py                    # Configuration settings
├── create_yaml.py               # Generate YOLO data.yaml
├── train.py                     # Training script with resume
├── evaluate.py                  # Model evaluation
├── visualize_predictions.py    # Prediction visualization
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── QUICKSTART.md               # Quick start guide
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Ultralytics YOLOv8

## Installation

```bash
cd yolo
pip install -r requirements.txt
```

## Dataset Structure

Your dataset is structured as follows:

```
dataset/
├── annotations/
│   ├── train.json       # COCO annotations (for Mask R-CNN)
│   ├── val.json         # COCO annotations (for Mask R-CNN)
│   └── test.json        # COCO annotations (for Mask R-CNN)
├── train/
│   ├── images/          # Training images (.tif format)
│   └── labels/          # YOLO .txt label files (0=ecoli, 1=salmonella)
├── val/
│   ├── images/          # Validation images (.tif format)
│   └── labels/          # YOLO .txt label files
└── test/
    ├── images/          # Test images (.tif format)
    └── labels/          # YOLO .txt label files
```

**Notes:**
- Images are in **TIF format** (.tif or .tiff)
- COCO annotations (`annotations/*.json`) are used for Mask R-CNN training
- YOLO labels (`train/labels/*.txt`, etc.) are used for YOLO training
- Both formats point to the same images

**YOLO Label Format:**
Each `.txt` file contains one line per object:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
- `class_id`: **0** for E. coli, **1** for Salmonella
- `x, y`: Normalized polygon coordinates (0-1)
- All coordinates for the segmentation mask

**Example label file (image_001.txt):**
```
0 0.123 0.234 0.145 0.256 0.167 0.278 0.189 0.290 0.123 0.234
1 0.456 0.567 0.478 0.589 0.490 0.601 0.512 0.623 0.456 0.567
0 0.789 0.890 0.801 0.912 0.823 0.934 0.845 0.956 0.789 0.890
```
This file indicates 2 E. coli instances (class 0) and 1 Salmonella instance (class 1).

## Configuration

Edit `config.py` to customize:
- Dataset paths
- Model selection (yolov8n/s/m/l/x-seg)
- Training hyperparameters
- Class names and mappings

### Default Configuration
- **Model**: yolov8m-seg (medium)
- **Classes**: 2
  - **Class 0**: E. coli (as labeled in your .txt files)
  - **Class 1**: Salmonella (as labeled in your .txt files)
- **Epochs**: 35
- **Batch Size**: 8
- **Image Size**: 1024
- **Save Period**: 5 epochs

**Important:** Your YOLO label files must use:
- `0` for E. coli instances
- `1` for Salmonella instances

## Quick Start Workflow

Your YOLO labels are already ready! Just 2 simple steps:

### Step 1: Create YAML Configuration

```bash
python create_yaml.py
```

This creates `data.yaml` pointing to your existing images and labels.

**Output:**
```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
nc: 2
names:
  - ecoli
  - salmonella
```

### Step 2: Train the Model

```bash
# Start training (or resume from checkpoint)
python train.py

# Start fresh training (ignore existing checkpoints)
python train.py --no-resume
```

**Training Features:**
✅ Automatic resume from `last.pt` if available
✅ Saves checkpoints every 5 epochs
✅ Saves best and last weights
✅ Generates training plots
✅ Validates during training
✅ Automatic test evaluation after training

**Checkpoints Saved:**
- `runs/yolo_segmentation/weights/best.pt` - Best model (highest mAP)
- `runs/yolo_segmentation/weights/last.pt` - Latest checkpoint
- `runs/yolo_segmentation/weights/epoch*.pt` - Periodic checkpoints

### Step 3: Evaluate the Model

```bash
# Evaluate on test set (default)
python evaluate.py

# Evaluate on validation set
python evaluate.py --split val

# Evaluate specific model
python evaluate.py --model runs/yolo_segmentation/weights/best.pt
```

**Metrics Provided:**
- Box mAP50, mAP50-95
- Segmentation mAP50, mAP50-95
- Precision and Recall
- Per-class metrics

### Step 4: Visualize Predictions

```bash
# Visualize single image
python visualize_predictions.py --images dataset/test/images/image1.jpg

# Visualize multiple images
python visualize_predictions.py --images image1.jpg image2.jpg image3.jpg

# Visualize all images in directory
python visualize_predictions.py --images dataset/test/images/

# Custom confidence threshold
python visualize_predictions.py \
    --images dataset/test/images/ \
    --conf 0.7 \
    --output_dir my_visualizations

# Use specific model
python visualize_predictions.py \
    --model runs/yolo_segmentation/weights/best.pt \
    --images dataset/test/images/
```

**Visualization Features:**
- Side-by-side: Original (left) vs Predictions (right)
- Color-coded masks: Red (E. coli), Blue (Salmonella)
- Bounding boxes with confidence scores
- Instance counts in title

## Resume Training

Training automatically resumes from the last checkpoint if available:

```bash
# This will automatically resume if last.pt exists
python train.py
```

The script will:
1. Look for `runs/yolo_segmentation/weights/last.pt`
2. If found, resume training from that checkpoint
3. If not found, start fresh training

To force fresh training:
```bash
python train.py --no-resume
```

## Model Selection

Edit `config.py` to change the model size:

```python
model_name = "yolov8n-seg"  # Nano (fastest, least accurate)
model_name = "yolov8s-seg"  # Small
model_name = "yolov8m-seg"  # Medium (default, balanced)
model_name = "yolov8l-seg"  # Large
model_name = "yolov8x-seg"  # Extra Large (slowest, most accurate)
```

## Advanced Options

### Custom Training Parameters

Edit `config.py`:

```python
num_epochs = 50          # Train for more epochs
batch_size = 16          # Larger batch (needs more GPU memory)
image_size = 1280        # Higher resolution
save_period = 10         # Save checkpoints less frequently
```

### Custom Class Mapping

If your COCO annotations use different class IDs:

```python
# In config.py
coco_to_yolo_mapping = {
    1: 0,  # COCO class 1 -> YOLO class 0 (ecoli)
    2: 1   # COCO class 2 -> YOLO class 1 (salmonella)
}
```

## Output Files

After training, you'll find:

```
runs/yolo_segmentation/
├── weights/
│   ├── best.pt         # Best model (use for inference)
│   ├── last.pt         # Latest checkpoint (for resuming)
│   └── epoch*.pt       # Periodic checkpoints
├── results.csv         # Training metrics
├── results.png         # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

## Troubleshooting

### CUDA Out of Memory
```python
# In config.py, reduce:
batch_size = 4  # or even 2
image_size = 640  # smaller image size
```

### Conversion Issues
If COCO to YOLO conversion fails, check:
- Annotation files are valid JSON
- Segmentation format is correct (polygon or RLE)
- Image dimensions are present in annotations

### Training Not Resuming
Make sure:
- `last.pt` exists in `runs/yolo_segmentation/weights/`
- Not using `--no-resume` flag
- `data.yaml` path is correct

## Comparison with Mask R-CNN

| Feature | YOLO | Mask R-CNN |
|---------|------|------------|
| Speed | ⚡ Faster | Slower |
| Accuracy | Good | Better |
| Training Time | Shorter | Longer |
| Real-time | ✅ Yes | ❌ No |
| Memory | Lower | Higher |
| Best For | Speed-critical applications | High-accuracy requirements |

## Citation

If you use YOLOv8, please cite:

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.


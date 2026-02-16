# YOLO Training - Quick Start Guide

## ⚡ Fast Setup (2 Commands!)

```bash
# 1. Create YAML config
python create_yaml.py

# 2. Start training
python train.py
```

That's it! Your model is training! 🎉

*(Assumes you already ran `pip install -r requirements.txt`)*

---

## 📋 Your Label Format

Your `.txt` files in `dataset/train/labels/`, `dataset/val/labels/`, `dataset/test/labels/` should contain:

```
0 x1 y1 x2 y2 x3 y3 ...  # E. coli (class 0)
1 x1 y1 x2 y2 x3 y3 ...  # Salmonella (class 1)
```

**Rules:**
- ✅ Class 0 = E. coli
- ✅ Class 1 = Salmonella
- ✅ Coordinates are normalized (0-1)
- ✅ Each line = one instance
- ✅ Filename matches image (image001.jpg → image001.txt)

---

## 🎯 Essential Commands

### Training
```bash
python train.py              # Auto-resumes from last.pt
python train.py --no-resume  # Start fresh
```

### Evaluation
```bash
python evaluate.py           # Test set
python evaluate.py --split val  # Validation set
```

### Visualization
```bash
python visualize_predictions.py --images dataset/test/images/
python visualize_predictions.py --images image1.jpg image2.jpg
python visualize_predictions.py --images test/ --conf 0.7
```

---

## 📂 Expected Directory Structure

```
your_project/
├── dataset/
│   ├── annotations/
│   │   ├── train.json     # COCO format
│   │   ├── val.json
│   │   └── test.json
│   ├── train/
│   │   ├── images/        # Your .tif images here
│   │   └── labels/        # Your .txt files here (0=ecoli, 1=salmonella)
│   ├── val/
│   │   ├── images/        # Your .tif images here
│   │   └── labels/        # Your .txt files here
│   └── test/
│       ├── images/        # Your .tif images here
│       └── labels/        # Your .txt files here
└── yolo/
    ├── config.py
    ├── create_yaml.py
    ├── train.py
    ├── evaluate.py
    └── visualize_predictions.py
```

**Image Format:** TIF (.tif or .tiff)  
**Label Format:** Text files with polygon coordinates

---

## ✅ Pre-Flight Checklist

Before running `python train.py`, verify:

- [ ] Images are in `dataset/{train,val,test}/images/`
- [ ] Labels are in `dataset/{train,val,test}/labels/`
- [ ] Label files use `0` for E. coli, `1` for Salmonella
- [ ] Label files have same names as images (except .txt extension)
- [ ] Ran `python create_yaml.py` successfully
- [ ] `data.yaml` file exists

---

## 🎓 Training Output

You'll see:
```
YOLO TRAINING CONFIGURATION
======================================================================
Dataset root: ../dataset
Model: yolov8m-seg
Number of classes: 2
  - Class 0: ecoli
  - Class 1: salmonella
Epochs: 35
Batch size: 8
======================================================================

STARTING YOLO TRAINING
======================================================================
Epoch 1/35: ...
```

---

## 💾 Where Your Models Are Saved

After training:
```
runs/yolo_segmentation/weights/
├── best.pt    ← Use this for inference (best performance)
├── last.pt    ← For resuming training
└── epoch*.pt  ← Checkpoints every 5 epochs
```

---

## 🔧 Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "CUDA out of memory"
Edit `config.py`:
```python
batch_size = 4  # or even 2
image_size = 640  # smaller size
```

### "No images found"
Check your paths in `config.py`:
```python
dataset_root = "../dataset"  # Adjust if needed
```

### Labels not matching
Verify each image has a corresponding .txt file:
```bash
# In dataset/train/
ls images/ | wc -l  # Count images
ls labels/ | wc -l  # Should match
```

---

## 📊 What to Expect

**Training time:** Depends on:
- GPU: ~2-5 hours (RTX 3090)
- Dataset size: Scales linearly
- Image size: 1024 is slower than 640

**Memory usage:**
- Batch 8: ~10-12GB VRAM
- Batch 4: ~6-8GB VRAM
- Batch 2: ~4-5GB VRAM

**Performance:**
- Should see mAP50 > 0.70 after 20-30 epochs
- Training loss should decrease steadily
- Validation mAP should stabilize

---

## 🚀 After Training

### Evaluate
```bash
python evaluate.py
```

### Visualize
```bash
python visualize_predictions.py --images dataset/test/images/
```

### Use in Production
```python
from ultralytics import YOLO

model = YOLO('runs/yolo_segmentation/weights/best.pt')
results = model.predict('your_image.jpg', conf=0.5)
```

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Verify your label format (0=ecoli, 1=salmonella)
3. Make sure `data.yaml` was created successfully
4. Check that labels directory exists and has .txt files

---

**Ready? Let's train!** 🎯

```bash
python create_yaml.py && python train.py
```

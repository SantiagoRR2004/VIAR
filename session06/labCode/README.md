# Lab 6: Object Detection - Setup and Instructions

## Overview

This lab implements two modern object detection architectures:
- **YOLO**: Anchor-based, grid-based detection
- **FCOS**: Anchor-free, per-pixel detection with FPN

Both models are trained and evaluated on the COCO dataset.

## Prerequisites

### Hardware Requirements
- **Minimum**: 8GB RAM, 6GB VRAM GPU (or CPU with patience)
- **Recommended**: 16GB RAM, 11GB+ VRAM GPU
- **Disk Space**: ~25GB for COCO dataset

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU support)

## Installation

### 1. Create Virtual Environment

```bash
# Create environment
python -m venv venv_lab6

# Activate (Linux/Mac)
source venv_lab6/bin/activate

# Activate (Windows)
venv_lab6\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install PyTorch (GPU version - CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pycocotools tensorboard matplotlib opencv-python tqdm numpy scipy
```

### 3. Download COCO Dataset

```bash
# Create data directory
mkdir -p data/coco
cd data/coco

# Download images (18GB train, 1GB val)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Clean up
rm *.zip

cd ../..
```

**Alternative**: Download subset for faster testing:
```bash
# Only download validation set (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## Project Structure

```
lab6_detection/
├── README.md                    # This file
├── lab6_complete.py             # Complete solution (for reference)
├── lab6_template.py             # Student template (your work goes here)
├── utils/
│   ├── visualization.py         # Visualization utilities
│   └── metrics.py               # Additional metrics
├── checkpoints/                 # Saved models
├── logs/                        # TensorBoard logs
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
```

## Implementation Roadmap

### Phase 1: Data Pipeline (2-3 hours)
**File**: `lab6_template.py`, sections 1.1-1.8

1. ✅ Implement `COCODetectionDataset.__init__()`
2. ✅ Implement `COCODetectionDataset.__getitem__()`
3. ✅ Implement `collate_fn()`
4. 🧪 **Test**: Run `test_dataset()`

**Expected output**:
```
Testing dataset...
Dataset size: 118287
Image shape: torch.Size([3, 448, 448])
Number of boxes: 7
✓ Dataset test passed!
```

### Phase 2: Utility Functions (1-2 hours)
**File**: `lab6_template.py`, sections 2.1-2.4

1. ✅ Implement `compute_iou()`
2. ✅ Implement `xywh_to_xyxy()` and `xyxy_to_xywh()`
3. ✅ Implement `nms()`
4. 🧪 **Test**: Run `test_iou()` and `test_nms()`

**Expected output**:
```
Testing IoU...
✓ IoU test passed!
Testing NMS...
✓ NMS test passed!
```

### Phase 3: YOLO Detector (3-4 hours)
**File**: `lab6_template.py`, sections 3.1-3.5

1. ✅ Implement `YOLODetector.__init__()` and `forward()`
2. ✅ Implement `YOLOLoss.encode_targets()`
3. ✅ Implement `YOLOLoss.get_responsible_boxes()`
4. ✅ Implement `YOLOLoss.forward()`
5. 🧪 **Test**: Run `test_yolo_model()`

**Expected output**:
```
Testing YOLO model...
✓ YOLO model test passed!
```

### Phase 4: FCOS Detector (3-4 hours)
**File**: `lab6_template.py`, sections 4.1-4.11

1. ✅ Implement `FPN.__init__()` and `forward()`
2. ✅ Implement `FCOSHead.__init__()` and `forward()`
3. ✅ Implement `FCOSDetector.__init__()` and `forward()`
4. ✅ Implement `FocalLoss.forward()`
5. ✅ Implement `FCOSLoss` methods
6. 🧪 **Test**: Run `test_fcos_model()`

**Expected output**:
```
Testing FCOS model...
✓ FCOS model test passed!
```

### Phase 5: Training & Evaluation (2-3 hours)
**File**: `lab6_template.py`, sections 5.1-5.4

1. ✅ Implement `train_one_epoch()`
2. ✅ Implement `evaluate_model()`
3. ✅ Implement `decode_yolo_predictions()`
4. ✅ Implement `decode_fcos_predictions()`

### Phase 6: Main Pipeline (1 hour)
**File**: `lab6_template.py`, section 6.1

1. ✅ Complete `main()` function
2. 🚀 **Start training**!

## Training

### Quick Start (YOLO)

```python
# In lab6_template.py, set:
model_type = 'yolo'
config.batch_size = 8
config.num_epochs = 20

# Run
python lab6_template.py
```

### Quick Start (FCOS)

```python
# In lab6_template.py, set:
model_type = 'fcos'
config.batch_size = 4  # FCOS needs more memory
config.num_epochs = 20

# Run
python lab6_template.py
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Open browser to http://localhost:6006
```

### Expected Training Time

**YOLO (ResNet18 backbone)**:
- GPU (RTX 3080): ~2 hours for 20 epochs
- GPU (GTX 1060): ~4 hours for 20 epochs
- CPU: ~30-40 hours for 20 epochs

**FCOS (ResNet50 backbone)**:
- GPU (RTX 3080): ~4 hours for 20 epochs
- GPU (GTX 1060): ~8 hours for 20 epochs
- CPU: Not recommended

### Tips for Faster Training

1. **Reduce dataset size** (for testing):
```python
# In lab6_template.py
train_dataset.img_ids = train_dataset.img_ids[:1000]  # Use only 1000 images
```

2. **Reduce batch size** if out of memory:
```python
config.batch_size = 2  # Minimum workable
```

3. **Use gradient accumulation**:
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Evaluation

### Run Evaluation

```python
from lab6_template import evaluate_model, FCOSDetector

# Load model
model = FCOSDetector(num_classes=80)
checkpoint = torch.load('checkpoints/fcos_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
metrics = evaluate_model(model, val_loader, val_dataset.coco, device)
print(f"mAP: {metrics['mAP']:.3f}")
print(f"AP50: {metrics['AP50']:.3f}")
print(f"AP75: {metrics['AP75']:.3f}")
```

### Visualize Predictions

```python
from utils.visualization import visualize_predictions

# Get predictions
images, boxes, labels, img_ids = next(iter(val_loader))
predictions = model(images.to(device))
detections = decode_fcos_predictions(predictions, ...)

# Visualize
visualize_predictions(images[0], detections[0], save_path='output.png')
```

## Expected Results

### YOLO (ResNet18, 20 epochs)
- **mAP**: 15-20%
- **AP50**: 30-35%
- **AP75**: 10-15%
- **Inference**: 30-40 FPS (GPU)

### FCOS (ResNet50, 20 epochs)
- **mAP**: 25-30%
- **AP50**: 42-48%
- **AP75**: 20-25%
- **APS**: 10-15% (small objects)
- **APM**: 25-30% (medium objects)
- **APL**: 35-40% (large objects)
- **Inference**: 15-20 FPS (GPU)

### Baseline Comparison
- **State-of-the-art YOLO (YOLOv8)**: 50-55% mAP
- **State-of-the-art FCOS**: 40-45% mAP

Our implementations achieve lower performance due to:
- Limited training time (20 epochs vs 300+)
- Smaller backbones (ResNet18/50 vs ResNet101/CSPDarknet)
- Basic augmentation (vs advanced techniques)
- No tricks (e.g., multi-scale training, test-time augmentation)

## Troubleshooting

### Issue: CUDA out of memory

**Solution**:
```python
# Reduce batch size
config.batch_size = 2

# Or use gradient accumulation
# Or train on CPU (slower)
config.device = 'cpu'
```

### Issue: Loss is NaN

**Possible causes**:
1. Learning rate too high → Reduce to 1e-5
2. Gradient explosion → Check gradient clipping
3. Division by zero → Check IoU and loss computations

**Debug**:
```python
# Add checks in loss computation
torch.autograd.set_detect_anomaly(True)
```

### Issue: mAP is 0% or very low (<5%)

**Possible causes**:
1. Incorrect target encoding → Check `encode_targets()`
2. Wrong box format (xywh vs xyxy) → Verify conversions
3. NMS too aggressive → Increase threshold to 0.7
4. Confidence threshold too high → Lower to 0.01

**Debug**:
```python
# Visualize targets vs predictions
from utils.visualization import plot_targets_and_predictions
plot_targets_and_predictions(images[0], targets, predictions)
```

### Issue: Training is too slow

**Solutions**:
1. Use smaller backbone (ResNet18 instead of ResNet50)
2. Reduce input size (320x320 instead of 448x448)
3. Use fewer FPN levels (only P4, P5)
4. Train on subset of data first

## Common Mistakes

### ❌ Mistake 1: Wrong box format
```python
# WRONG: Mixing formats
boxes_xywh = [0.5, 0.5, 0.2, 0.2]
iou = compute_iou(boxes_xywh, boxes_xyxy)  # Different formats!

# CORRECT: Consistent format
boxes_xyxy1 = [0.4, 0.4, 0.6, 0.6]
boxes_xyxy2 = [0.45, 0.45, 0.65, 0.65]
iou = compute_iou(boxes_xyxy1, boxes_xyxy2, box_format='xyxy')
```

### ❌ Mistake 2: Not normalizing boxes
```python
# WRONG: Absolute pixel coordinates
boxes = [100, 100, 200, 200]

# CORRECT: Normalized [0, 1]
boxes = [100/448, 100/448, 200/448, 200/448]
```

### ❌ Mistake 3: Forgetting to set model mode
```python
# WRONG: No mode set
predictions = model(images)

# CORRECT: Set appropriate mode
model.train()  # For training
predictions = model(images)

model.eval()  # For evaluation
with torch.no_grad():
    predictions = model(images)
```

### ❌ Mistake 4: Not clipping gradients
```python
# WRONG: Can lead to gradient explosion
loss.backward()
optimizer.step()

# CORRECT: Clip gradients
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
optimizer.step()
```

## Deliverables

### Required Files

1. **Code** (`lab6_template.py`):
   - All TODOs completed
   - Code runs without errors
   - Passes all tests

2. **Trained Models**:
   - `checkpoints/yolo_best.pth`
   - `checkpoints/fcos_best.pth`
   - Training logs

3. **Report** (PDF, 5-10 pages):
   - **Introduction**: Brief overview of YOLO and FCOS
   - **Implementation**: Key design decisions and challenges
   - **Results**: 
     - Training curves (loss over epochs)
     - Evaluation metrics table
     - Example predictions (5-10 images)
   - **Analysis**:
     - YOLO vs FCOS comparison
     - Scale-specific performance (APS, APM, APL)
     - Failure cases and why
   - **Conclusion**: Lessons learned and potential improvements

### Grading Rubric (100 points)

**Code (60 points)**:
- Dataset implementation: 10 pts
- Utility functions (IoU, NMS): 5 pts
- YOLO model and loss: 15 pts
- FCOS model and loss: 20 pts
- Training and evaluation: 10 pts

**Model Performance (20 points)**:
- YOLO mAP ≥ 15%: 8 pts (partial: 10-15% → 4 pts)
- FCOS mAP ≥ 25%: 12 pts (partial: 20-25% → 6 pts)

**Report (20 points)**:
- Completeness and clarity: 8 pts
- Results presentation: 6 pts
- Analysis depth: 6 pts

**Bonus (up to 10% extra credit)**:
- Implement Soft-NMS: +2%
- Implement DIoU/CIoU loss: +3%
- Advanced augmentation experiments: +2%
- Multi-scale training: +3%

## Resources

### Papers
- **YOLO**: [You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640)
- **FCOS**: [FCOS: Fully Convolutional One-Stage (Tian et al., 2019)](https://arxiv.org/abs/1904.01355)
- **FPN**: [Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144)
- **Focal Loss**: [Focal Loss for Dense Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)

### Documentation
- [PyTorch Docs](https://pytorch.org/docs/)
- [COCO API](https://github.com/cocodataset/cocoapi)
- [Torchvision](https://pytorch.org/vision/stable/index.html)

### Reference Implementations
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## Support

**Office Hours**: [Your office hours]
**Forum**: [Your course forum]
**Email**: [Your email]

## License

This lab material is for educational purposes only.
COCO dataset: [COCO License](https://cocodataset.org/#termsofuse)

---

**Good luck! 🚀**

Remember: Start early, test incrementally, and ask for help when stuck!
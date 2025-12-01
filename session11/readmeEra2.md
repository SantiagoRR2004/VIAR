# Era II Demos: Early Deep Learning for 3D (2015-2016)

## Overview

These demos implement the two foundational methods that brought deep learning to 3D shape understanding:

1. **3D ShapeNets** (Wu et al., 2015) - Voxel-based 3D CNNs
2. **Multi-View CNN** (Su et al., 2015) - Multi-view aggregation with 2D CNNs

Both represent different approaches to the fundamental challenge: **how do we apply deep learning to 3D data?**

---

## Method 1: 3D ShapeNets (Wu et al., 2015)

### Core Idea

Represent 3D shapes as **voxel grids** and process them with **3D convolutional neural networks**.

**Key Innovation:** First end-to-end learned 3D representation using deep learning.

### Architecture

```
Input: 32×32×32 voxel grid
    ↓
3D Conv (6×6×6, stride 2) → 16×16×16, 48 channels
    ↓
3D Conv (5×5×5, stride 2) → 8×8×8, 160 channels
    ↓
3D Conv (4×4×4, stride 2) → 4×4×4, 512 channels
    ↓
Flatten → 32,768 features
    ↓
FC (2048) → FC (1024) → FC (num_classes)
```

### Implementation: `3d_shapenets_demo.py`

**Features:**
- Complete 3D CNN implementation with 3D convolutions
- Two tasks: classification and shape completion
- Encoder-decoder architecture for completion
- Synthetic voxel generation for multiple shape types
- 3D visualization of voxel grids
- Shape completion visualization (partial → complete)

**Tasks:**

#### Task 1: 3D Shape Classification

Classify 3D objects from voxel representation.

```bash
python 3d_shapenets_demo.py \
    --task classification \
    --resolution 32 \
    --num_classes 10 \
    --epochs 50 \
    --batch_size 16
```

**Expected Results:**
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Training time: ~1-2 hours (32³ voxels)

#### Task 2: 3D Shape Completion

Complete partial 3D scans.

```bash
python 3d_shapenets_demo.py \
    --task completion \
    --resolution 32 \
    --occlusion_ratio 0.3 \
    --epochs 50 \
    --batch_size 16
```

**Expected Results:**
- Completion loss decreases steadily
- Visual quality improves over training
- Can complete 30% occluded shapes

### Mathematical Foundation

**3D Convolution:**
```
(w * v)[i,j,k] = Σ Σ Σ w[a,b,c] · v[i+a, j+b, k+c]
                 a b c
```

**Memory Complexity:** O(n³ · C) where n is resolution, C is channels

**Computational Complexity:** O(n³ · C_in · C_out · k³) per layer

### Advantages

✓ Direct 3D processing
✓ Preserves spatial structure
✓ End-to-end differentiable
✓ Can handle shape completion

### Limitations

✗ O(n³) memory scaling limits resolution
✗ Typically limited to 32³ or 64³ voxels
✗ Cannot represent fine details
✗ Expensive 3D convolutions

### Synthetic Shapes Generated

The demo generates various geometric primitives:
- **Cube** (class 0): Solid rectangular prism
- **Sphere** (class 1): Spherical shape
- **Cylinder** (class 2): Vertical cylinder
- **Pyramid** (class 3): Square pyramid
- **Torus** (class 4): Donut shape
- **Custom shapes** (classes 5-9): Random blobs

---

## Method 2: Multi-View CNN (Su et al., 2015)

### Core Idea

**Render 3D objects from multiple viewpoints** and leverage powerful **2D CNNs** to extract features from each view, then **aggregate** via pooling.

**Key Innovation:** Leverages ImageNet pre-trained 2D CNNs for 3D understanding.

### Architecture

```
3D Object
    ↓
Render V views (e.g., V=12)
    ↓
For each view:
    2D CNN (ResNet-18) → features_v
    ↓
View pooling (max/mean/learned)
    ↓
Aggregated features
    ↓
FC layers → Classification
```

### Implementation: `multiview_cnn_demo.py`

**Features:**
- Complete multi-view rendering pipeline
- Pre-trained ResNet-18 as feature extractor
- Three pooling strategies: max, mean, learned attention
- Simple geometric mesh rendering
- Multi-view visualization
- Camera positioning on sphere

**Usage:**

```bash
python multiview_cnn_demo.py \
    --num_views 12 \
    --pooling max \
    --pretrained \
    --epochs 30 \
    --batch_size 8
```

**Pooling Strategies:**

1. **Max Pooling** (default):
   ```python
   features_3d = max(features_view1, ..., features_viewV)
   ```
   Takes maximum activation across all views.

2. **Mean Pooling**:
   ```python
   features_3d = mean(features_view1, ..., features_viewV)
   ```
   Averages features across views.

3. **Learned Attention**:
   ```python
   weights = softmax(MLP(features))
   features_3d = Σ weights_v · features_v
   ```
   Learns to weight important views.

**Expected Results:**
- Training accuracy: ~90-95%
- Validation accuracy: ~85-90%
- Better than 3D ShapeNets on classification
- Training time: ~2-3 hours

### Mathematical Foundation

**View Feature Extraction:**
```
f_v = CNN_2D(I_v)  for v = 1, ..., V
```

**View Pooling:**
```
f_3D = pool(f_1, f_2, ..., f_V)
```

where pool ∈ {max, mean, learned}

**Classification:**
```
y = MLP(f_3D)
```

### Advantages

✓ Leverages powerful pre-trained 2D CNNs
✓ Higher effective resolution than voxels
✓ State-of-the-art accuracy (2015)
✓ Efficient: uses 2D convolutions
✓ Transfer learning from ImageNet

### Limitations

✗ Requires rendering multiple views
✗ Loses explicit 3D structure
✗ Hard to generate new 3D shapes
✗ Rendering step not differentiable
✗ More views = more computation

### Camera Configuration

By default, 12 views are rendered from:
- **4 views** at elevation 0° (equator)
- **4 views** at elevation 30° (above)
- **4 views** at elevation -30° (below)

Each set distributed evenly in azimuth (0°, 90°, 180°, 270°).

---

## Comparison: 3D ShapeNets vs Multi-View CNN

| Aspect | 3D ShapeNets | Multi-View CNN |
|--------|--------------|----------------|
| **Representation** | Voxel grid | Multiple 2D images |
| **Processing** | 3D convolutions | 2D convolutions |
| **Resolution** | Low (32³ typical) | High (224×224 per view) |
| **Memory** | O(n³) - expensive | O(V × H × W) - cheaper |
| **Pre-training** | None (train from scratch) | ImageNet transfer learning |
| **Accuracy** | ~75-80% (ModelNet40) | ~89-90% (ModelNet40) |
| **Generation** | Possible (encoder-decoder) | Difficult |
| **3D Structure** | Explicit | Implicit |

---

## Installation

### Required Packages

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
matplotlib>=3.5.0
scipy>=1.7.0
Pillow>=9.0.0
tqdm>=4.60.0
```

### Installation Steps

```bash
# Create environment
conda create -n era2_demos python=3.9
conda activate era2_demos

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy matplotlib scipy Pillow tqdm
```

---

## Usage Examples

### Example 1: Quick 3D Classification

```bash
# 3D ShapeNets
python 3d_shapenets_demo.py --task classification --epochs 20

# Multi-View CNN
python multiview_cnn_demo.py --epochs 20
```

### Example 2: Shape Completion

```bash
python 3d_shapenets_demo.py \
    --task completion \
    --resolution 32 \
    --occlusion_ratio 0.4 \
    --epochs 50
```

### Example 3: Ablation Study - Pooling Strategies

```bash
# Max pooling
python multiview_cnn_demo.py --pooling max --epochs 30

# Mean pooling
python multiview_cnn_demo.py --pooling mean --epochs 30

# Learned attention
python multiview_cnn_demo.py --pooling learned --epochs 30
```

### Example 4: Resolution Comparison

```bash
# Low resolution (fast)
python 3d_shapenets_demo.py --resolution 24 --epochs 30

# Standard resolution
python 3d_shapenets_demo.py --resolution 32 --epochs 50

# High resolution (slow, memory intensive)
python 3d_shapenets_demo.py --resolution 64 --epochs 50
```

---

## Outputs

### 3D ShapeNets

**Classification:**
- `shapenets_classifier.pth` - Trained model
- `shapenets_example.png` - 3D voxel visualization
- Console output with accuracy metrics

**Completion:**
- `shapenets_completion.pth` - Trained model
- `completion_results.png` - Before/after comparison
- Console output with loss metrics

### Multi-View CNN

- `multiview_cnn_best.pth` - Trained model checkpoint
- `multiview_example.png` - Grid of rendered views
- Console output with accuracy metrics

---

## Key Parameters

### 3D ShapeNets

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | classification | Task: classification or completion |
| `--resolution` | 32 | Voxel grid resolution (24, 32, 64) |
| `--num_classes` | 10 | Number of shape categories |
| `--occlusion_ratio` | 0.3 | Fraction of voxels removed (completion) |
| `--batch_size` | 16 | Batch size |
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-3 | Learning rate |

### Multi-View CNN

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_views` | 12 | Number of rendered views |
| `--pooling` | max | Pooling strategy (max/mean/learned) |
| `--pretrained` | True | Use ImageNet pre-trained weights |
| `--image_size` | 224 | Size of rendered images |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 30 | Training epochs |
| `--lr` | 1e-4 | Learning rate |

---

## Memory Requirements

### 3D ShapeNets

| Resolution | Memory (per sample) | Batch Size | GPU Memory |
|------------|---------------------|------------|------------|
| 24³ | ~55 KB | 32 | ~4 GB |
| 32³ | ~130 KB | 16 | ~4 GB |
| 64³ | ~1 MB | 4 | ~6 GB |

### Multi-View CNN

| Num Views | Image Size | Batch Size | GPU Memory |
|-----------|------------|------------|------------|
| 8 | 224×224 | 16 | ~4 GB |
| 12 | 224×224 | 8 | ~4 GB |
| 20 | 224×224 | 4 | ~4 GB |

---

## Troubleshooting

### 3D ShapeNets

**Issue: CUDA out of memory**
```bash
# Reduce resolution
--resolution 24

# Reduce batch size
--batch_size 8

# Reduce number of channels (modify code)
```

**Issue: Slow training**
```bash
# Use lower resolution
--resolution 24

# Reduce number of samples
--num_train 500
```

### Multi-View CNN

**Issue: CUDA out of memory**
```bash
# Reduce number of views
--num_views 8

# Reduce batch size
--batch_size 4

# Use smaller images
--image_size 128
```

**Issue: Poor accuracy**
```bash
# Use pre-trained weights
--pretrained

# Increase number of views
--num_views 20

# Train longer
--epochs 50
```

---

## Extensions and Exercises

### Exercise 1: Resolution Study (3D ShapeNets)
Compare classification accuracy at different resolutions:
- 16³, 24³, 32³, 48³, 64³
- Plot accuracy vs resolution
- Plot memory usage vs resolution
- Analyze the trade-off

### Exercise 2: View Count Study (Multi-View CNN)
Investigate effect of number of views:
- Test with 4, 8, 12, 16, 20 views
- Plot accuracy vs number of views
- Find the optimal number
- Visualize which views are most important

### Exercise 3: Hybrid Approach
Combine both methods:
- Extract voxel features with 3D ShapeNets
- Extract multi-view features with MVCNN
- Concatenate and classify
- Compare with individual methods

### Exercise 4: Real Data
Apply to real datasets:
- Download ModelNet40 dataset
- Implement proper data loading
- Compare with reported results
- Analyze failure cases

---

## Historical Context

### Why These Papers Matter

1. **3D ShapeNets** showed that 3D CNNs could learn meaningful representations
   - First to use deep learning directly on 3D data
   - Introduced the voxel representation paradigm
   - Enabled probabilistic shape completion

2. **Multi-View CNN** achieved state-of-the-art by being clever
   - Leveraged existing 2D CNN advances
   - Transfer learning from ImageNet
   - Showed that sometimes indirect approaches work better

### Impact

- **3D ShapeNets** inspired: VoxNet, 3D U-Net, V-Net, 3D ResNet
- **Multi-View CNN** inspired: MVCNN variants, RotationNet, view-based methods
- Both established baselines for ModelNet and ShapeNet benchmarks
- Led to the question: "Can we do better?" → PointNet (2017)

---

## References

### Original Papers

1. **3D ShapeNets:**
   - Wu et al., "3D ShapeNets: A Deep Representation for Volumetric Shapes", CVPR 2015
   - [Paper](http://3dshapenets.cs.princeton.edu/)

2. **Multi-View CNN:**
   - Su et al., "Multi-view Convolutional Neural Networks for 3D Shape Recognition", ICCV 2015
   - [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf)

### Datasets

- **ModelNet40**: 12,311 CAD models, 40 categories
  - [Download](https://modelnet.cs.princeton.edu/)
- **ModelNet10**: Subset with 10 categories
- **ShapeNet**: Large-scale 3D dataset
  - [Website](https://shapenet.org/)

### Related Work

- VoxNet (2015): 3D CNNs for real-time object recognition
- 3D-GAN (2016): Generative models for 3D shapes
- RotationNet (2018): Multi-view with rotation prediction
- PointNet (2017): Direct point cloud processing (next era!)

---

## Performance Benchmarks

### ModelNet40 (Reported Results)

| Method | Accuracy | Year |
|--------|----------|------|
| 3D ShapeNets | 77.0% | 2015 |
| VoxNet | 83.0% | 2015 |
| Multi-View CNN | 90.1% | 2015 |
| PointNet | 89.2% | 2017 |
| PointNet++ | 91.9% | 2017 |

*Note: Our demos use synthetic data, so accuracy will differ*

---

## Citation

If you use these implementations for research, please cite the original papers:

```bibtex
@inproceedings{wu2015shapenets,
  title={3D ShapeNets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{su2015multiview,
  title={Multi-view convolutional neural networks for 3D shape recognition},
  author={Su, Hang and Maji, Subhransu and Kalogerakis, Evangelos and Learned-Miller, Erik},
  booktitle={ICCV},
  year={2015}
}
```

---

**Last Updated:** November 2024  
**Version:** 1.0  
**Course:** VIAR25/26, Universidad de Vigo  
**Instructor:** Prof. David Olivieri
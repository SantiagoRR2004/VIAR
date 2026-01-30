# Lecture 11: 3D Scene Understanding and Neural Rendering

## Overview

This package contains complete materials for Lecture 11 on 3D Scene Understanding and Neural Rendering, covering the evolution from classical multi-view geometry through point cloud networks to modern neural radiance fields.

## Contents

### 1. Lecture Slides (`lec11_v2.tex`)

Comprehensive LaTeX Beamer presentation with:

**Chronological Structure:**

- **Era I (1990-2014):** Classical 3D Reconstruction (Structure from Motion, Multi-View Stereo)
- **Era II (2015-2016):** Early Deep Learning for 3D (3D ShapeNets, Multi-View CNN)
- **Era III (2017-2019):** Point Cloud Networks (PointNet, PointNet++, PointPillars)
- **Era IV (2020-2022):** Neural Rendering Revolution (NeRF, Instant-NGP, TensoRF)
- **Era V (2023-2024):** 3D Foundation Models (Gaussian Splatting, DreamFusion, LRM)

**Features:**

- Detailed timeline visualization showing method evolution
- Conceptual TikZ diagrams for 3D representations and architectures
- Algorithm2e pseudocode for key methods
- Mathematical foundations and derivations
- Comprehensive references to Python demos

**Key Topics:**

- 3D representations (point clouds, meshes, voxels, implicit functions)
- PointNet architecture and permutation invariance
- NeRF and volume rendering
- 3D object detection
- Real-time neural rendering (Gaussian Splatting)
- Generative 3D models (text-to-3D)

### 2. Python Demos

Three complete, well-documented implementations:

#### `pointnet_demo.py` - PointNet Classification

**What it does:**

- Implements complete PointNet architecture for 3D shape classification
- Includes T-Net for spatial transformation
- Demonstrates permutation-invariant learning on point clouds

**Key concepts:**

- Shared MLP for per-point features
- Max-pooling for global feature aggregation
- Feature transform regularization
- Data augmentation for point clouds

**Usage:**

```bash
python pointnet_demo.py --num_points 1024 --epochs 50 --batch_size 32
```

**Expected results:**

- Training accuracy: ~91%
- Validation accuracy: ~87%
- Training time: ~2 hours on single GPU

**Learning objectives:**

- Understand how to process unordered point sets
- Learn about permutation invariance in deep learning
- Implement spatial transformation networks
- Visualize learned features in point cloud space

---

#### `nerf_demo.py` - Neural Radiance Fields

**What it does:**

- Implements simplified NeRF for novel view synthesis
- Includes positional encoding and volume rendering
- Demonstrates implicit neural scene representations

**Key concepts:**

- Continuous 5D scene representation (position + direction → color + density)
- Positional encoding for high-frequency details
- Volume rendering via numerical integration
- Hierarchical sampling for efficiency

**Usage:**

```bash
python nerf_demo.py --n_iters 10000 --batch_size 4096 --n_samples 64
```

**Expected results:**

- Training PSNR: ~31 dB
- Rendering time: ~1 second per frame
- High-quality novel view synthesis

**Learning objectives:**

- Understand implicit neural representations
- Learn volume rendering mathematics
- Implement differentiable rendering
- Master positional encoding techniques

---

#### `pointpillars_demo.py` - 3D Object Detection

**What it does:**

- Implements PointPillars for 3D object detection from LiDAR
- Converts point clouds to bird's eye view representation
- Predicts 3D bounding boxes with orientation

**Key concepts:**

- Pillar-based point cloud encoding
- PointNet feature extraction per pillar
- 2D convolution on BEV pseudo-image
- Multi-task detection head (classification + regression + direction)

**Usage:**

```bash
python pointpillars_demo.py --num_samples 100 --epochs 10
```

**Expected results:**

- Inference: ~50ms per frame
- Real-time capable for autonomous driving
- 3D IoU-based evaluation

**Learning objectives:**

- Understand 3D object detection pipeline
- Learn bird's eye view representations
- Implement multi-head detection networks
- Master 3D bounding box parameterization

## Prerequisites

### Required Packages

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
tqdm>=4.60.0

# For visualization (optional)
open3d>=0.15.0  # For point cloud visualization
imageio>=2.9.0  # For video generation
```

### Installation

```bash
# Create conda environment
conda create -n viar_lec11 python=3.9
conda activate viar_lec11

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy matplotlib tqdm
pip install open3d imageio  # Optional
```

## Running the Demos

### 1. PointNet Demo

**Quick start:**

```bash
python pointnet_demo.py --epochs 50
```

**Advanced usage:**

```bash
python pointnet_demo.py \
    --num_points 1024 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --dropout 0.3
```

**Output:**

- `pointnet_best.pth` - Trained model checkpoint
- Console output with training progress and metrics

---

### 2. NeRF Demo

**Quick start:**

```bash
python nerf_demo.py --n_iters 10000
```

**Advanced usage:**

```bash
python nerf_demo.py \
    --num_images 100 \
    --img_size 100 \
    --n_iters 10000 \
    --batch_size 1024 \
    --n_samples 64 \
    --pos_L 10 \
    --dir_L 4
```

**Output:**

- `nerf_model.pth` - Trained NeRF model
- `nerf_results.png` - Novel view synthesis results
- Console output with PSNR metrics

---

### 3. PointPillars Demo

**Quick start:**

```bash
python pointpillars_demo.py --epochs 10
```

**Advanced usage:**

```bash
python pointpillars_demo.py \
    --num_samples 100 \
    --epochs 10 \
    --lr 2e-4
```

**Output:**

- `pointpillars_model.pth` - Trained detector
- Console output with loss metrics

## Compiling the Slides

### Requirements

- Full LaTeX distribution (TeX Live or MiKTeX)
- Required packages: beamer, tikz, algorithm2e, listings

### Compilation

```bash
# Standard compilation
pdflatex lec11_v2.tex
pdflatex lec11_v2.tex  # Run twice for references

# With notes
pdflatex -interaction=nonstopmode lec11_v2.tex
```

## Lab Exercises

### Exercise 1: PointNet Extensions

**Tasks:**

1. Implement PointNet++ hierarchical sampling
2. Add part segmentation capability
3. Visualize critical points that determine classification
4. Experiment with different symmetric functions (mean, sum vs max)

**Difficulty:** Medium
**Time:** 3-4 hours

---

### Exercise 2: NeRF Improvements

**Tasks:**

1. Implement hierarchical sampling with fine network
2. Add appearance embedding for varying lighting
3. Optimize rendering speed with caching
4. Experiment with different positional encoding frequencies

**Difficulty:** Hard
**Time:** 4-6 hours

---

### Exercise 3: 3D Detection Pipeline

**Tasks:**

1. Implement 3D IoU calculation
2. Add non-maximum suppression for 3D boxes
3. Visualize detections on point clouds
4. Evaluate on metrics: precision, recall, AP

**Difficulty:** Hard
**Time:** 4-6 hours

---

### Bonus Exercise: Gaussian Splatting

**Tasks:**

1. Use pre-trained 3D Gaussian Splatting code
2. Capture your own scene with photos
3. Train and render in real-time
4. Compare quality and speed with NeRF

**Difficulty:** Medium
**Time:** 2-3 hours

## Key Concepts Summary

### Point Cloud Processing

- **Permutation invariance:** Order shouldn't matter
- **Max-pooling:** Symmetric function for global features
- **T-Net:** Learned canonical alignment
- **Hierarchical sampling:** Multi-scale feature extraction

### Neural Rendering

- **Implicit representation:** Continuous function f(x,d) → (c,σ)
- **Volume rendering:** Integrate along rays
- **Positional encoding:** Enable high-frequency details
- **Hierarchical sampling:** Concentrate samples where density is high

### 3D Detection

- **Bird's eye view:** Project to top-down representation
- **Pillar encoding:** Group points vertically
- **Multi-task learning:** Classification + regression + direction
- **3D IoU:** Volumetric intersection over union

## Additional Resources

### Papers

1. **PointNet** (Qi et al., 2017): "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
2. **NeRF** (Mildenhall et al., 2020): "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
3. **PointPillars** (Lang et al., 2019): "PointPillars: Fast Encoders for Object Detection from Point Clouds"
4. **Instant-NGP** (Müller et al., 2022): "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
5. **3D Gaussian Splatting** (Kerbl et al., 2023): "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

### Datasets

- **ModelNet40:** 3D shape classification (12,311 CAD models, 40 categories)
- **ShapeNet:** Large-scale 3D shape repository (51,300 models, 55 categories)
- **KITTI:** Autonomous driving with LiDAR (7,481 training, 7,518 test)
- **NeRF Synthetic:** Photorealistic rendering (8 objects, known camera poses)
- **Matterport3D:** Indoor scenes with RGB-D (10,800 panoramas, 90 buildings)

### Code Repositories

- PointNet: https://github.com/charlesq34/pointnet
- NeRF: https://github.com/bmild/nerf
- PointPillars: https://github.com/nutonomy/second.pytorch
- Instant-NGP: https://github.com/NVlabs/instant-ngp
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting

## Troubleshooting

### Common Issues

**1. CUDA out of memory**

- Reduce batch size: `--batch_size 16` (PointNet) or `--batch_size 512` (NeRF)
- Reduce number of samples: `--n_samples 32` (NeRF)
- Use smaller hidden dimensions: `--hidden_dim 128`

**2. Slow training**

- Enable GPU if available: PyTorch will automatically use CUDA
- Use DataLoader with num_workers > 0 for parallel data loading
- Enable mixed precision training (advanced)

**3. Poor convergence**

- Check learning rate: try `--lr 5e-4` or `--lr 1e-3`
- Increase training iterations/epochs
- Check data normalization and preprocessing
- Verify gradient flow (no NaN or inf values)

**4. LaTeX compilation errors**

- Install missing packages: `tlmgr install <package>`
- Update TeX distribution to latest version
- Check for special characters in paths

## Assessment

Students will be evaluated on:

1. **Implementation (40%)**
   - Code correctness and completeness
   - Proper use of PyTorch conventions
   - Documentation and code clarity

2. **Understanding (30%)**
   - Written explanations of key concepts
   - Ability to modify and extend implementations
   - Answers to conceptual questions

3. **Experimentation (20%)**
   - Hyperparameter tuning results
   - Ablation studies
   - Analysis of failure cases

4. **Presentation (10%)**
   - Results visualization
   - Clear communication of findings
   - Comparison with baselines

## Contact

For questions about the lecture materials or implementations:

- Instructor: Prof. David Olivieri
- Office Hours: [to be announced]
- Email: [contact information]
- Course Forum: [link if available]

## License

These educational materials are provided for use in VIAR25/26 at Universidad de Vigo.

## Acknowledgments

This lecture builds upon foundational work by:

- Charles Qi (PointNet)
- Ben Mildenhall (NeRF)
- Alex Lang (PointPillars)
- The broader computer vision and graphics research community

---

**Last updated:** November 2024
**Version:** 2.0

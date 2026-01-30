# Classical 3D Reconstruction Demos

This package contains two educational demos illustrating classical (pre-deep learning) 3D reconstruction methods covered in the VIAR course.

## Demos Overview

### 1. Structure from Motion (SfM) - `demo_sfm_classical.py`

**Purpose**: Demonstrates the complete SfM pipeline from 2D feature correspondences to sparse 3D reconstruction.

**Pipeline Steps**:

1. **Feature Detection & Matching**: Establishes 2D correspondences between views
2. **Fundamental Matrix Estimation**: Uses RANSAC + 8-point algorithm to estimate epipolar geometry
3. **Essential Matrix & Pose Recovery**: Extracts relative camera pose (R, t)
4. **Triangulation**: Reconstructs 3D points from 2D correspondences
5. **Bundle Adjustment**: Jointly optimizes camera poses and 3D points

**Key Concepts Illustrated**:

- Epipolar geometry: x₂ᵀ F x₁ = 0
- RANSAC for robust estimation
- Direct Linear Transform (DLT) for triangulation
- Non-linear optimization (Levenberg-Marquardt)
- Scale ambiguity in monocular reconstruction

**Output**:

- Visualization showing feature matches, inliers/outliers
- 3D sparse point cloud reconstruction
- Ground truth vs reconstructed comparison
- Quantitative error metrics before/after bundle adjustment

---

### 2. Multi-View Stereo (MVS) - `demo_mvs_classical.py`

**Purpose**: Demonstrates dense reconstruction from sparse SfM using photometric consistency.

**Pipeline Steps**:

1. **Scene Generation**: Creates textured 3D surface with depth variation
2. **Sparse SfM Simulation**: Generates sparse point cloud (simulating SfM output)
3. **Cost Volume Construction**: Builds 3D cost volume using NCC matching
4. **Depth Map Estimation**: Per-pixel depth selection via winner-takes-all
5. **Depth Fusion**: Merges multiple depth maps into dense point cloud

**Key Concepts Illustrated**:

- Normalized Cross-Correlation (NCC) for photometric consistency
- Cost volume aggregation across multiple views
- Depth hypothesis sampling
- Confidence-based filtering
- Dense reconstruction (10-100x more points than SfM)

**Output**:

- Rendered images from multiple viewpoints
- Per-view depth maps (color-coded)
- Confidence maps showing matching quality
- Sparse vs dense 3D comparison
- Densification statistics

---

## Installation & Requirements

### Dependencies

```bash
pip install numpy matplotlib opencv-python scipy
```

**Required packages**:

- `numpy` - Numerical computations
- `matplotlib` - Visualization and 3D plotting
- `opencv-python` (cv2) - Rodrigues conversion, image operations
- `scipy` - Optimization (least_squares), interpolation

### Python Version

- Python 3.7+

---

## Running the Demos

### Demo 1: Structure from Motion

```bash
python demo_sfm_classical.py
```

**Expected output**:

```
======================================================================
CLASSICAL STRUCTURE FROM MOTION (SfM) DEMO
======================================================================

[Step 1] Generating synthetic 3D scene...
  - Created 58 3D points

[Step 2] Setting up cameras...
  - Camera intrinsics K:
    [[800.   0. 320.]
     [  0. 800. 240.]
     [  0.   0.   1.]]

[Step 3] Projecting 3D points to 2D images...
  - Camera 1: 58 points
  - Camera 2: 58 points

[Step 4] Estimating Fundamental Matrix with RANSAC...
  - Found 52 / 58 inliers

[Step 5] Recovering camera pose from Essential Matrix...

[Step 6] Triangulating 3D points...
  - Reconstructed 52 3D points
  - Mean reconstruction error: 0.0234

[Step 7] Running Bundle Adjustment...
  - Mean error after BA: 0.0156
  - Improvement: 0.0078

[Step 8] Visualizing results...
```

**Runtime**: ~5-10 seconds

---

### Demo 2: Multi-View Stereo

```bash
python demo_mvs_classical.py
```

**Expected output**:

```
======================================================================
CLASSICAL MULTI-VIEW STEREO (MVS) DEMO
======================================================================

[Step 1] Generating synthetic textured scene...
  - Created 19200 3D points

[Step 2] Setting up camera array...
  - Created 3 cameras

[Step 3] Rendering images from each camera...
  - Camera 1: rendered 320x240 image
  - Camera 2: rendered 320x240 image
  - Camera 3: rendered 320x240 image

[Step 4] Simulating sparse SfM output...
  - Sparse SfM: 960 points

[Step 5] Computing depth maps using MVS...
  Camera 1 as reference...
    Computing depth map (240 x 320) with 32 depth samples...
    Mean depth error: 0.1234

[Step 6] Fusing depth maps into dense point cloud...
  - Total dense points: 12456

Reconstruction Statistics:
  - Sparse SfM points: 960
  - Dense MVS points: 12456
  - Densification ratio: 13.0x
```

**Runtime**: ~30-60 seconds (depending on CPU)

---

## Understanding the Outputs

### SfM Demo Visualization (`sfm_demo_results.png`)

The output figure contains 6 subplots:

1. **Top-Left**: Camera 1 feature points (blue dots)
2. **Top-Center**: Camera 2 feature points (red dots)
3. **Top-Right**: Feature matches (green = inliers, red = outliers)
4. **Bottom-Left**: Ground truth 3D scene with camera poses
5. **Bottom-Center**: SfM reconstruction before bundle adjustment
6. **Bottom-Right**: Overlay comparison (GT vs reconstructed)

**What to observe**:

- RANSAC successfully identifies inliers (typically 80-95% of matches)
- Triangulated 3D points approximate ground truth
- Bundle adjustment reduces reprojection errors
- Camera poses (black triangles) are recovered correctly

---

### MVS Demo Visualization (`mvs_demo_results.png`)

The output figure contains 4 rows:

**Row 1**: Input images from 3 cameras (grayscale textured)
**Row 2**: Depth maps (color-coded, jet colormap)

- Red = close to camera
- Blue = far from camera

**Row 3**: Confidence maps (NCC scores)

- Yellow/Green = high confidence (good texture match)
- Purple/Blue = low confidence (ambiguous or textureless)

**Row 4**: 3D reconstructions

- Left: Sparse SfM (~1000 points)
- Center: Dense MVS (~10,000-15,000 points)
- Right: Overlay comparison

**What to observe**:

- Depth maps capture surface geometry with depth variations
- Confidence is high in textured regions, low in uniform areas
- Dense reconstruction fills in most visible surface points
- MVS produces 10-100x more points than sparse SfM

---

## Educational Goals

### Learning Objectives

After running these demos, students should understand:

1. **SfM Principles**:
   - How 2D correspondences constrain 3D geometry (epipolar constraint)
   - Why RANSAC is essential for robust estimation
   - The role of triangulation in recovering depth
   - How bundle adjustment refines the entire reconstruction
   - Scale ambiguity in monocular vision

2. **MVS Principles**:
   - Photometric consistency as a matching criterion
   - Cost volume construction and aggregation
   - Depth hypothesis testing
   - The difference between sparse (SfM) and dense (MVS) reconstruction
   - Why texture is crucial for stereo matching

3. **Classical vs Deep Learning Methods**:
   - Classical methods are interpretable and geometrically principled
   - Hand-crafted features (SIFT) vs learned features (CNNs)
   - Explicit optimization vs end-to-end learning
   - When classical methods fail (textureless, reflective, thin structures)

---

## Customization & Experimentation

### Experiment Ideas

**SfM Demo**:

1. **Add more noise**: Increase `noise_std` in `add_noise_to_points()`
2. **Add outliers**: Manually corrupt some correspondences
3. **Change camera baseline**: Modify translation `t2` (larger baseline = better depth accuracy)
4. **Vary number of points**: Change `n_points` in `generate_synthetic_scene()`

**MVS Demo**:

1. **Change depth sampling**: Modify `depth_samples` (trade-off: accuracy vs speed)
2. **Adjust patch size**: Change `patch_size` (larger = more robust, but smooths details)
3. **Test different textures**: Modify `create_synthetic_texture()` to add/remove patterns
4. **Add more cameras**: Extend the camera array for better reconstruction

### Key Parameters to Tune

**SfM**:

```python
# RANSAC parameters
threshold=3.0          # Inlier threshold (pixels)
n_iterations=1000      # RANSAC iterations

# Noise level
noise_std=0.5          # Gaussian noise (pixels)

# Camera baseline
t2 = np.array([1.5, 0, 0])  # Larger = better depth, but more occlusions
```

**MVS**:

```python
# Depth range
depth_min=3.0, depth_max=7.0    # Scene depth bounds

# Resolution
depth_samples=32                 # Number of depth hypotheses

# Matching
patch_size=7                     # NCC window size
confidence_threshold=0.3         # Minimum NCC to accept point
```

---

## Limitations of Classical Methods

These demos also illustrate common failure modes:

### SfM Failures:

- **Textureless surfaces**: No features to match (walls, sky)
- **Repetitive patterns**: Ambiguous correspondences
- **Dynamic scenes**: Moving objects violate static assumption
- **Specular surfaces**: Violate Lambertian reflectance

### MVS Failures:

- **Uniform regions**: NCC is ambiguous without texture
- **Thin structures**: Missed by discrete depth sampling
- **Occlusions**: Foreground/background confusion
- **View-dependent appearance**: Specular highlights, reflections

**Solution in modern methods**: Deep learning (CNNs) can learn robust features and matching functions that handle these cases better.

---

## Connection to Course Material

These demos correspond to **Section 1: Classical 3D Reconstruction (Era I: 1990-2014)** in the lecture slides.

### Slide References:

**SfM Demo** covers:

- "Structure from Motion: Epipolar Geometry and Optimization"
- "Structure from Motion: End-to-End Pipeline"
- Algorithms: 8-point algorithm, RANSAC, bundle adjustment

**MVS Demo** covers:

- "Classical 3D: Multi-View Stereo (MVS)"
- "Multi-View Stereo: Depth Maps, Fusion, and Failure Modes"
- "Multi-View Stereo: From Sparse SfM to Dense Geometry"

### Recommended Reading:

- Hartley & Zisserman: "Multiple View Geometry in Computer Vision"
- Szeliski: "Computer Vision: Algorithms and Applications" (Chapter 11)
- COLMAP: Schönberger & Frahm, "Structure-from-Motion Revisited" (CVPR 2016)

---

## Troubleshooting

### Common Issues:

**1. Import errors**:

```
ModuleNotFoundError: No module named 'cv2'
```

**Solution**: Install OpenCV: `pip install opencv-python`

**2. Slow execution (MVS demo)**:

```
Computing depth map takes > 5 minutes
```

**Solution**:

- Reduce `depth_samples` (e.g., 16 instead of 32)
- Increase `step` in depth map computation (line 239)

**3. Poor reconstruction quality**:

- Check that `noise_std` is not too high
- Ensure sufficient texture in synthetic scene
- Verify camera baseline is appropriate for depth range

**4. Visualization issues**:

```
UserWarning: Matplotlib is currently using agg...
```

**Solution**: This is normal for headless systems. Plots are still saved to disk.

---

## Credits & References

**Author**: Demo for VIAR Course (Artificial Vision)  
**Institution**: Universidad de Vigo, Computer Science Department  
**Course**: VIAR - Visión Artificial

**References**:

1. Hartley & Zisserman (2003) - Multi-view geometry fundamentals
2. Schönberger et al. (2016) - COLMAP SfM system
3. Furukawa & Ponce (2010) - Patch-based MVS
4. Szeliski (2022) - Computer Vision textbook

---

## Next Steps

After understanding these classical methods, explore:

1. **Era II (2015-2018)**: Deep learning for stereo (DispNet, GC-Net, PSM-Net)
2. **Era III (2018-2020)**: Neural implicit representations (NeRF, DeepSDF)
3. **Era IV (2020+)**: Gaussian splatting, diffusion-based 3D generation

These classical foundations are still relevant:

- COLMAP remains the gold standard for SfM benchmarking
- MVS principles inspire modern cost volume designs (e.g., in MVSNet)
- Bundle adjustment is used to refine neural 3D representations

---

## License

Educational use only. Code provided for VIAR course students.

# Quick Start Guide: Classical 3D Reconstruction Demos

## 📦 Package Contents

You have received 6 files for classical 3D reconstruction demonstrations:

| File                        | Purpose                    | Runtime |
| --------------------------- | -------------------------- | ------- |
| `test_installation.py`      | Verify dependencies        | 5 sec   |
| `demo_sfm_classical.py`     | Structure from Motion demo | 10 sec  |
| `demo_mvs_classical.py`     | Multi-View Stereo demo     | 60 sec  |
| `demo_complete_pipeline.py` | SfM → MVS pipeline         | 10 sec  |
| `README_classical_demos.md` | Detailed documentation     | -       |
| `README_QUICKSTART.md`      | This file                  | -       |

---

## 🚀 Quick Start (3 steps)

### Step 1: Install Dependencies

```bash
pip install numpy matplotlib opencv-python scipy
```

### Step 2: Verify Installation

```bash
python test_installation.py
```

Expected output:

```
✓ numpy           v1.24.3      - Numerical computation
✓ matplotlib      v3.7.1       - Visualization
✓ cv2             v4.8.0       - OpenCV (computer vision)
✓ scipy           v1.11.1      - Scientific computing
ALL TESTS PASSED! ✓
```

### Step 3: Run Demos

```bash
# Individual demos (recommended order)
python demo_sfm_classical.py        # Structure from Motion
python demo_mvs_classical.py        # Multi-View Stereo

# Or run complete pipeline
python demo_complete_pipeline.py    # SfM → MVS pipeline
```

---

## 📊 What Each Demo Shows

### Demo 1: Structure from Motion (`demo_sfm_classical.py`)

**What it does**: Reconstructs sparse 3D structure from 2D image correspondences

**Key techniques**:

- ✓ Feature matching
- ✓ Fundamental matrix (RANSAC)
- ✓ Camera pose recovery
- ✓ 3D triangulation
- ✓ Bundle adjustment

**Output**: `sfm_demo_results.png`

- Feature correspondences (inliers vs outliers)
- Sparse 3D point cloud
- Ground truth comparison
- Error metrics

**Runtime**: ~5-10 seconds

---

### Demo 2: Multi-View Stereo (`demo_mvs_classical.py`)

**What it does**: Creates dense 3D reconstruction using photometric consistency

**Key techniques**:

- ✓ Cost volume construction
- ✓ Normalized Cross-Correlation (NCC)
- ✓ Depth map estimation
- ✓ Multi-view fusion
- ✓ Confidence filtering

**Output**: `mvs_demo_results.png`

- Input images from multiple cameras
- Per-view depth maps (color-coded)
- Confidence maps (NCC scores)
- Sparse vs dense comparison

**Runtime**: ~30-60 seconds

---

### Demo 3: Complete Pipeline (`demo_complete_pipeline.py`)

**What it does**: Shows the full SfM → MVS workflow

**Highlights**:

- ✓ Camera array setup
- ✓ Sparse reconstruction (SfM)
- ✓ Dense reconstruction (MVS)
- ✓ Densification analysis
- ✓ Multiple viewing angles

**Output**: `complete_pipeline_comparison.png`

- Multi-panel comprehensive visualization
- Point density comparison
- Pipeline diagram
- Statistics panel

**Runtime**: ~10 seconds

---

## 🎯 Recommended Learning Path

### For First-Time Users:

1. **Start here**: `demo_complete_pipeline.py`
   - Gets the big picture
   - Understands workflow
   - Sees SfM vs MVS comparison

2. **Deep dive SfM**: `demo_sfm_classical.py`
   - Understands epipolar geometry
   - Learns triangulation
   - Sees bundle adjustment in action

3. **Deep dive MVS**: `demo_mvs_classical.py`
   - Understands photometric consistency
   - Learns depth estimation
   - Sees dense reconstruction

4. **Read details**: `README_classical_demos.md`
   - Full technical documentation
   - Customization guide
   - Troubleshooting tips

---

## 🔧 Common Issues & Solutions

### Issue 1: Import Error

**Problem**:

```
ModuleNotFoundError: No module named 'cv2'
```

**Solution**:

```bash
pip install opencv-python
```

---

### Issue 2: Slow Execution (MVS demo)

**Problem**: MVS demo takes > 5 minutes

**Solution**: Edit `demo_mvs_classical.py`, line 239:

```python
# Change this:
depth_samples=32

# To this (faster, less accurate):
depth_samples=16
```

---

### Issue 3: No Display Window

**Problem**: Plots don't show up

**Solution**: This is normal on headless systems. Check for saved PNG files:

- `sfm_demo_results.png`
- `mvs_demo_results.png`
- `complete_pipeline_comparison.png`

---

## 📖 Understanding the Outputs

### SfM Output Interpretation

**Look for**:

- **High inlier ratio** (>80%): Good feature matching
- **Low reconstruction error** (<0.05): Accurate triangulation
- **Improvement after BA**: Bundle adjustment is working

**Red flags**:

- Low inlier ratio (<50%): Poor feature matching
- High error (>0.5): Camera calibration issues

---

### MVS Output Interpretation

**Look for**:

- **Smooth depth maps**: Good photometric consistency
- **High confidence** (green regions): Textured areas
- **Dense coverage**: 10-20× more points than SfM

**Red flags**:

- Noisy depth maps: Insufficient texture
- Low confidence (purple): Uniform or specular regions
- Holes in reconstruction: Occlusions or thin structures

---

## 🎓 Educational Value

### Learning Objectives

After running these demos, you should understand:

**Technical Concepts**:

1. ✓ Epipolar geometry and fundamental matrix
2. ✓ Camera pose estimation from correspondences
3. ✓ 3D triangulation from multiple views
4. ✓ Bundle adjustment for refinement
5. ✓ Photometric consistency for dense matching
6. ✓ Depth map fusion strategies

**Practical Insights**:

1. ✓ When classical methods work well (textured, static scenes)
2. ✓ When they fail (reflections, repetitive patterns)
3. ✓ Trade-offs: speed vs accuracy vs density
4. ✓ Why deep learning helps (learned features, end-to-end optimization)

---

## 🔬 Experimentation Ideas

### Easy Experiments (5-10 minutes each):

1. **Noise sensitivity**: Increase `noise_std` in SfM demo
2. **Baseline effects**: Modify camera translation in SfM
3. **Depth resolution**: Change `depth_samples` in MVS
4. **Patch size**: Modify `patch_size` in MVS

### Advanced Experiments (30-60 minutes each):

1. **Add more cameras**: Extend camera array in MVS
2. **Failure cases**: Create textureless regions
3. **Real images**: Adapt demos for real photo input
4. **Optimization**: Profile and optimize bottlenecks

---

## 📚 Further Reading

### Essential Papers:

1. **SfM**: Schönberger & Frahm (2016) - "Structure-from-Motion Revisited"
2. **MVS**: Furukawa & Ponce (2010) - "Accurate, Dense, and Robust Multiview Stereopsis"
3. **Bundle Adjustment**: Triggs et al. (2000) - "Bundle Adjustment — A Modern Synthesis"

### Textbooks:

1. Hartley & Zisserman (2003) - "Multiple View Geometry in Computer Vision"
2. Szeliski (2022) - "Computer Vision: Algorithms and Applications"

### Software:

1. **COLMAP**: State-of-the-art open-source SfM/MVS system
2. **OpenMVG**: Open Multiple View Geometry library
3. **AliceVision**: Open-source photogrammetry framework

---

## 💡 Tips for Success

### Running Demos:

1. ✓ Always run `test_installation.py` first
2. ✓ Start with `demo_complete_pipeline.py` for overview
3. ✓ Close previous plot windows before running new demos
4. ✓ Check output PNG files if plots don't display
5. ✓ Reduce `depth_samples` if MVS is too slow

### Understanding Outputs:

1. ✓ Read axis labels carefully (units, scales)
2. ✓ Compare sparse vs dense point counts
3. ✓ Check error metrics (mean, median)
4. ✓ Observe confidence patterns in MVS
5. ✓ Notice camera positions in 3D views

### Experimentation:

1. ✓ Change ONE parameter at a time
2. ✓ Document your changes
3. ✓ Compare results quantitatively
4. ✓ Understand why changes affect output
5. ✓ Try to break the demos (learn failure modes)

---

## 🆘 Getting Help

### Troubleshooting Checklist:

- [ ] Dependencies installed? (`test_installation.py`)
- [ ] Correct Python version? (3.7+)
- [ ] In correct directory? (all files present)
- [ ] Sufficient RAM? (>2GB recommended)
- [ ] Output files generated? (check PNG files)

### Common Error Messages:

| Error                        | Cause              | Solution                |
| ---------------------------- | ------------------ | ----------------------- |
| `ModuleNotFoundError`        | Missing package    | `pip install <package>` |
| `MemoryError`                | Insufficient RAM   | Reduce point counts     |
| `UserWarning: Matplotlib...` | Headless system    | Check PNG files         |
| Optimization not converging  | Bad initialization | Check random seed       |

---

## 📝 Summary

### What You Have:

✓ Three complete, runnable demos  
✓ Comprehensive documentation  
✓ Test suite for verification  
✓ Real implementation of classical algorithms  
✓ Educational visualizations

### What You'll Learn:

✓ How SfM recovers 3D from 2D matches  
✓ How MVS creates dense reconstructions  
✓ When classical methods work/fail  
✓ Foundation for modern deep learning methods  
✓ Practical 3D reconstruction pipeline

### Next Steps:

1. ✓ Run all three demos
2. ✓ Read detailed README
3. ✓ Experiment with parameters
4. ✓ Compare with lecture slides
5. ✓ Explore modern alternatives (NeRF, Gaussian Splatting)

---

## 📧 Course Information

**Course**: VIAR - Visión Artificial (Artificial Vision)  
**Institution**: Universidad de Vigo, Computer Science Department  
**Level**: Advanced undergraduate / Graduate

**Related Topics**:

- Era I (1990-2014): Classical methods (these demos)
- Era II (2015-2018): Deep stereo (DispNet, PSM-Net)
- Era III (2018-2020): Neural representations (NeRF)
- Era IV (2020+): Gaussian splatting, diffusion models

---

## ✅ Success Criteria

You've successfully completed the demos if you can:

- [ ] Run all three demos without errors
- [ ] Explain the SfM pipeline steps
- [ ] Explain the MVS pipeline steps
- [ ] Interpret the output visualizations
- [ ] Identify when methods would fail
- [ ] Modify parameters and observe effects
- [ ] Connect concepts to lecture material

**Congratulations!** You now understand classical 3D reconstruction! 🎉

---

_For detailed technical documentation, see `README_classical_demos.md`_  
_For troubleshooting, run `test_installation.py`_

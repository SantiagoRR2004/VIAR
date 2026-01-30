# Two-Stream Convolutional Networks for Action Recognition

**Paper**: Simonyan & Zisserman, NIPS 2014  
**Link**: https://arxiv.org/abs/1406.2199

## Overview

Two-Stream Networks pioneered the approach of using separate pathways for spatial (appearance) and temporal (motion) information in video understanding. This method was one of the first to effectively recognize actions in videos by combining information from both RGB frames and optical flow.

## Key Concepts

### 1. Dual Stream Architecture

The network consists of two independent CNNs:
- **Spatial Stream**: Processes single RGB frames to recognize objects and scenes
- **Temporal Stream**: Processes stacked optical flow fields to capture motion patterns

### 2. Optical Flow

Optical flow represents the motion between consecutive frames:
- Computed using methods like TV-L1 or Farneback
- Stacked across multiple frames (e.g., 10 frames = 20 channels for x and y flow)
- Captures dense motion information across the entire frame

### 3. Late Fusion

The two streams are combined at the decision level:
- Each stream produces class scores independently
- Final prediction averages the softmax scores from both streams
- Alternative: weighted fusion or other combination strategies

## Architecture Details

### Spatial Stream
- **Input**: Single RGB frame (3 channels, 224×224)
- **Backbone**: Modified VGG-16 architecture
- **Pre-training**: ImageNet for better initialization

### Temporal Stream
- **Input**: Stacked optical flow (20 channels, 224×224)
- **Backbone**: VGG-16 style architecture
- **Training**: From scratch on action recognition data

### Network Structure
Input RGB [3, 224, 224] → VGG-16 → FC layers → Spatial Scores [num_classes]
Input Flow [20, 224, 224] → VGG-16 → FC layers → Temporal Scores [num_classes]
Final Prediction = Average(Spatial Scores, Temporal Scores)

## Key Innovations

1. **Separation of Concerns**: Explicitly modeling appearance and motion separately
2. **Optical Flow as Input**: Using pre-computed motion as a separate modality
3. **Transfer Learning**: Leveraging ImageNet pre-training for the spatial stream
4. **Simple Fusion**: Effective late fusion strategy

## Strengths

- Clear separation between spatial and temporal modeling
- Strong baseline for action recognition
- Can leverage ImageNet pre-training
- Interpretable: can see what each stream learns

## Limitations

- Requires expensive optical flow computation
- Two separate networks increase computational cost
- Limited temporal modeling (only local motion)
- No end-to-end optimization across streams

## Training Strategy

1. **Spatial Stream**:
   - Initialize with ImageNet pre-trained weights
   - Fine-tune on action recognition dataset
   - Sample individual frames during training

2. **Temporal Stream**:
   - Train from scratch
   - Sample short optical flow stacks
   - Data augmentation: random cropping, flipping

3. **Fusion**:
   - Test-time fusion of predictions
   - Can weight streams based on validation performance

## Impact

Two-Stream Networks established the foundation for many subsequent video understanding methods:
- Inspired temporal segment networks (TSN)
- Led to research on better fusion strategies
- Motivated end-to-end learnable motion features (e.g., 3D CNNs)

## Usage
```python
from twostream_demo import TwoStreamNetwork

# Create model
model = TwoStreamNetwork(num_classes=101)

# Prepare inputs
rgb_frame = torch.randn(1, 3, 224, 224)  # Single RGB frame
optical_flow = torch.randn(1, 20, 224, 224)  # 10 flows × 2 directions

# Forward pass
scores = model(rgb_frame, optical_flow, fusion='late')
```

## References

1. Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action recognition in videos. NIPS.
2. Wang, L., et al. (2016). Temporal segment networks. ECCV. (Follow-up work)
3. Carreira, J., & Zisserman, A. (2017). Quo vadis, action recognition? CVPR. (Comparison study)
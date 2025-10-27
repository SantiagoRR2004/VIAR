"""
Two-Stream Convolutional Networks for Action Recognition
Simonyan & Zisserman, NIPS 2014

This demo implements the Two-Stream architecture that processes
RGB frames and optical flow separately, then fuses predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TwoStreamNetwork(nn.Module):
    """Two-Stream Network with separate spatial and temporal pathways."""
    
    def __init__(self, num_classes: int = 101, dropout: float = 0.5):
        super().__init__()
        self.spatial_stream = self._build_stream(input_channels=3, num_classes=num_classes)
        self.temporal_stream = self._build_stream(input_channels=20, num_classes=num_classes)
        self.dropout = dropout
        
    def _build_stream(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build a single stream using modified VGG architecture"""
        layers = []
        
        # Conv blocks
        layers.extend([
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        for _ in range(2):
            layers.extend([
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        
        stream = nn.Sequential(*layers)
        
        stream.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, num_classes)
        )
        
        return stream
    
    def forward(self, rgb: torch.Tensor, flow: torch.Tensor, fusion: str = 'late') -> torch.Tensor:
        """
        Forward pass with late fusion.
        
        Args:
            rgb: RGB frame [B, 3, H, W]
            flow: Optical flow (stacked) [B, 20, H, W]
            fusion: Type of fusion ('late')
        
        Returns:
            Fused class scores [B, num_classes]
        """
        if fusion == 'late':
            spatial_out = self.spatial_stream(rgb)
            spatial_scores = self.spatial_stream.classifier(spatial_out)
            
            temporal_out = self.temporal_stream(flow)
            temporal_scores = self.temporal_stream.classifier(temporal_out)
            
            return (spatial_scores + temporal_scores) / 2
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")


def main():
    """Demo of Two-Stream Networks"""
    print("="*80)
    print("Two-Stream Networks Demo")
    print("Simonyan & Zisserman, NIPS 2014")
    print("="*80)
    
    # Create model
    model = TwoStreamNetwork(num_classes=101)
    
    # Simulate input: video with 11 frames
    video = torch.randn(2, 11, 3, 224, 224)
    
    # Extract middle frame for spatial stream
    rgb_frame = video[:, 5]  # [2, 3, 224, 224]
    
    # Simulate optical flow (10 flows × 2 directions = 20 channels)
    flow = torch.randn(2, 20, 224, 224)
    
    print(f"\nInput shapes:")
    print(f"  RGB frame: {rgb_frame.shape}")
    print(f"  Optical flow: {flow.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        scores = model(rgb_frame, flow, fusion='late')
        probs = F.softmax(scores, dim=1)
        
        print(f"\nOutput shape: {scores.shape}")
        print(f"\nTop-5 predictions for sample 1:")
        top5 = torch.topk(probs[0], k=5)
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"  {i+1}. Class {idx.item()}: {prob.item()*100:.2f}%")
    
    print("\n" + "="*80)
    print("Key Insight: Separate spatial and temporal processing")
    print("- Spatial stream: Recognizes objects from single RGB frames")
    print("- Temporal stream: Captures motion from optical flow")
    print("- Late fusion: Average predictions from both streams")
    print("="*80)


if __name__ == '__main__':
    main()
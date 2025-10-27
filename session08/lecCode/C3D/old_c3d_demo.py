"""
Learning Spatiotemporal Features with 3D Convolutional Networks (C3D)
Tran et al., ICCV 2015

This demo implements C3D, which uses 3D convolutions to jointly model
spatial and temporal dimensions in videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class C3D(nn.Module):
    """C3D: 3D Convolutional Network for video classification."""
    
    def __init__(self, num_classes: int = 101, dropout: float = 0.5):
        super().__init__()
        
        # Layer 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Layer 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Layer 3
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Layer 4
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Layer 5
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Fully connected layers
        self.fc6 = nn.Linear(512 * 4 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input video clips [B, C, T, H, W]
               where T=16, H=W=112
        
        Returns:
            Class scores [B, num_classes]
        """
        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        
        # Flatten
        x = x.flatten(1)
        
        # FC layers
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = self.fc8(x)
        
        return x


def main():
    """Demo of C3D"""
    print("="*80)
    print("C3D: 3D Convolutional Networks Demo")
    print("Tran et al., ICCV 2015")
    print("="*80)
    
    # Create model
    model = C3D(num_classes=101)
    
    # Create sample input: 16-frame clips at 112x112 resolution
    video = torch.randn(2, 3, 16, 112, 112)
    
    print(f"\nInput shape: {video.shape}")
    print(f"  [batch_size, channels, temporal, height, width]")
    print(f"  Temporal dimension: 16 frames")
    print(f"  Spatial resolution: 112×112")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(video)
        probs = F.softmax(logits, dim=1)
        
        print(f"\nOutput shape: {logits.shape}")
        print(f"\nTop-5 predictions for sample 1:")
        top5 = torch.topk(probs[0], k=5)
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"  {i+1}. Class {idx.item()}: {prob.item()*100:.2f}%")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print("Key Insight: 3D convolutions for spatiotemporal learning")
    print("- 3×3×3 kernels capture both spatial and temporal patterns")
    print("- Hierarchical features from low-level motion to high-level actions")
    print("- End-to-end trainable without optical flow")
    print("="*80)


if __name__ == '__main__':
    main()
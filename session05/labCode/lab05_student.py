"""
Lab 5: Advanced Image Segmentation
Student Template - Complete all TODOs

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================== Part 1: FCN Architecture ==================
def make_bilinear_weights(kernel_size, num_channels):
    """Create weights for ConvTranspose2d to behave like bilinear upsampling.

    Esto aplica la inicialización de He/Kaiming, que es una forma estándar de arrancar pesos en redes con ReLU.
    En lugar de valores aleatorios cualquiera, distribuye los pesos según la
    varianza adecuada para que las activaciones ni exploten ni se apaguen.
    Esto ayuda a que la red empiece a aprender bien desde el primer paso.
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(kernel_size).unsqueeze(0)
    filt = 1 - torch.abs(og - center) / factor
    weight = filt.t() * filt
    weight = weight / weight.sum()
    # Create a (out_c, in_c, k, k) tensor where each (c,c) has the kernel, others zero
    w = torch.zeros((num_channels, num_channels, kernel_size, kernel_size))
    for i in range(num_channels):
        w[i, i] = weight
    return w


class FCN32s(nn.Module):
    """FCN without skip connections (baseline)"""

    def __init__(self, n_classes=21, kaiming_weight_init=True):
        super().__init__()

        # TODO Task 1.1: Load pretrained ResNet50 and extract layers

        # 1. Load models.resnet50(pretrained=True)
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1, layer2, layer3, layer4
        resnet = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
        # remove the last 2 layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # TODO Task 1.2: Add score layer
        # 1. Create 1x1 convolution: nn.Conv2d(2048, n_classes, kernel_size=1)
        self.conv1x1 = nn.Conv2d(2048, n_classes, kernel_size=1)

        # TODO Task 1.3: Add upsampling layer
        # 1. Create transposed convolution for 32x upsampling
        # 2. Use nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, bias=False)
        self.upsample32x = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=64, stride=32, bias=False
        )

        if kaiming_weight_init:
            # Initialize the score conv (kaiming) and upsample to bilinear init
            nn.init.kaiming_normal_(
                self.score.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.score.bias is not None:
                nn.init.constant_(self.score.bias, 0)

            # init upsample to approximate bilinear interpolation
            with torch.no_grad():
                w = make_bilinear_weights(64, n_classes)
                self.upsample32.weight.copy_(w)

    def forward(self, x):
        # TODO Task 1.4: Implement forward pass
        # 1. Pass through conv1, bn1, relu, maxpool
        # 2. Pass through layer1, layer2, layer3, layer4
        # 3. Apply score layer (1x1 conv)
        # 4. Apply 32x upsampling
        # 5. Return output

        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.upsample32x(x)
        return x


class FCN16s(nn.Module):
    """FCN with one skip connection from pool4"""

    def __init__(self, n_classes=21):
        super().__init__()

        # TODO Task 1.1: Load pretrained ResNet50 and extract layers
        # (Same as FCN32s)

        # TODO Task 1.2: Add score layers
        # 1. score_pool4: nn.Conv2d(1024, n_classes, 1) for layer3 output
        # 2. score_fr: nn.Conv2d(2048, n_classes, 1) for layer4 output

        # TODO Task 1.3: Add upsampling layers
        # 1. upscore2: 2x upsampling with stride=2
        # 2. upscore16: 16x upsampling with stride=16

        pass

    def forward(self, x):
        # TODO Task 1.4: Implement forward pass with one skip connection
        # 1. Pass through initial layers until layer3 (save as pool4)
        # 2. Continue to layer4
        # 3. Apply score_fr to layer4 output
        # 4. Apply score_pool4 to pool4
        # 5. Upsample score_fr by 2x (use upscore2)
        # 6. Add upsampled score_fr + score_pool4 (element-wise addition)
        # 7. Upsample fused result by 16x
        # 8. Return output

        pass


class FCN8s(nn.Module):
    """Fully Convolutional Network with two skip connections"""

    def __init__(self, n_classes=21):
        super().__init__()

        # TODO Task 1.1: Load pretrained ResNet50 and extract layers
        # 1. Load models.resnet50(pretrained=True)
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (stride 4)
        # 4. Extract layer2 (stride 8, will be pool3)
        # 5. Extract layer3 (stride 16, will be pool4)
        # 6. Extract layer4 (stride 32)

        # TODO Task 1.2: Add score layers (1x1 convolutions)
        # 1. score_pool3: nn.Conv2d(512, n_classes, 1) - layer2 outputs 512 channels
        # 2. score_pool4: nn.Conv2d(1024, n_classes, 1) - layer3 outputs 1024 channels
        # 3. score_fr: nn.Conv2d(2048, n_classes, 1) - layer4 outputs 2048 channels

        # TODO Task 1.3: Add upsampling layers
        # 1. upscore2: nn.ConvTranspose2d for 2x upsampling (32 -> 16)
        # 2. upscore_pool4: nn.ConvTranspose2d for 2x upsampling (16 -> 8)
        # 3. upscore8: nn.ConvTranspose2d for 8x upsampling (8 -> 1)
        # All should have: (n_classes, n_classes, kernel_size=4 or 16, stride=2 or 8, bias=False)

        pass

    def forward(self, x):
        # TODO Task 1.4: Implement forward pass with progressive skip fusion
        # ENCODER PATH:
        # 1. x = relu(bn1(conv1(x)))
        # 2. x = maxpool(x)
        # 3. x = layer1(x)
        # 4. pool3 = layer2(x) - Save for skip connection
        # 5. pool4 = layer3(pool3) - Save for skip connection
        # 6. x = layer4(pool4)

        # SCORE LAYERS:
        # 7. score_fr = score_fr(x)
        # 8. score_pool4 = score_pool4(pool4)
        # 9. score_pool3 = score_pool3(pool3)

        # PROGRESSIVE UPSAMPLING WITH SKIP CONNECTIONS:
        # First skip (pool4 at stride 16):
        # 10. upscore2 = upscore2(score_fr) - Upsample 32 -> 16
        # 11. If shapes don't match, use F.interpolate to resize
        # 12. fuse_pool4 = upscore2 + score_pool4 - Element-wise addition

        # Second skip (pool3 at stride 8):
        # 13. upscore_pool4 = upscore_pool4(fuse_pool4) - Upsample 16 -> 8
        # 14. If shapes don't match, use F.interpolate to resize
        # 15. fuse_pool3 = upscore_pool4 + score_pool3 - Element-wise addition

        # Final upsampling:
        # 16. out = upscore8(fuse_pool3) - Upsample 8 -> 1 (original resolution)
        # 17. Return out

        pass


# ================== Part 2: DeepLabV3+ Architecture ==================


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""

    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()

        # TODO Task 2.1: Implement Branch 1 - 1x1 convolution
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)

        # TODO Task 2.1: Implement Branches 2-4 - Atrous convolutions
        # 1. Create nn.ModuleList()
        # 2. For each rate in rates [6, 12, 18]:
        #    - Append nn.Sequential with:
        #      * nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False)
        #      * nn.BatchNorm2d(out_channels)
        #      * nn.ReLU(inplace=True)

        # TODO Task 2.1: Implement Branch 5 - Global average pooling
        # 1. Create nn.Sequential with:
        #    - nn.AdaptiveAvgPool2d(1) - Pools to 1x1
        #    - nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)

        # TODO Task 2.2: Implement fusion layer
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)
        #    - nn.Dropout(0.1)

        pass

    def forward(self, x):
        # TODO Task 2.2: Apply all branches and concatenate
        # 1. Save spatial dimensions: size = x.shape[2:]

        # Branch 1:
        # 2. feat1 = self.conv1(x)

        # Branches 2-4:
        # 3. feat_atrous = [conv(x) for conv in self.atrous_convs]

        # Branch 5:
        # 4. feat_global = self.global_avg_pool(x)
        # 5. Interpolate feat_global back to original size using F.interpolate

        # Concatenation:
        # 6. feat = torch.cat([feat1] + feat_atrous + [feat_global], dim=1)

        # Output projection:
        # 7. return self.conv_out(feat)

        pass


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ architecture with ASPP"""

    def __init__(self, n_classes=21, backbone="resnet50"):
        super().__init__()

        # TODO Task 2.3: Load backbone and extract encoder layers
        # 1. Load models.resnet50(pretrained=True)
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (low-level features, stride 4)
        # 4. Extract layer2, layer3, layer4

        # TODO Task 2.3: Create ASPP module
        # 1. self.aspp = ASPP(2048, 256) - ResNet50 layer4 outputs 2048 channels

        # TODO Task 2.3: Create low-level feature projection
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(256, 48, 1, bias=False) - layer1 outputs 256 channels
        #    - nn.BatchNorm2d(48)
        #    - nn.ReLU(inplace=True)

        # TODO Task 2.3: Create decoder
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False) - ASPP (256) + low-level (48)
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU(inplace=True)
        #    - nn.Conv2d(256, 256, 3, padding=1, bias=False)
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU(inplace=True)

        # TODO Task 2.3: Create classification head
        # 1. nn.Conv2d(256, n_classes, 1)

        pass

    def forward(self, x):
        # TODO Task 2.4: Implement forward pass
        # 1. Save input size: size = x.shape[2:]

        # ENCODER:
        # 2. x = relu(bn1(conv1(x)))
        # 3. x = maxpool(x)
        # 4. low_level_feat = layer1(x) - Save for decoder
        # 5. x = layer2(low_level_feat)
        # 6. x = layer3(x)
        # 7. x = layer4(x)

        # ASPP:
        # 8. x = self.aspp(x)

        # DECODER:
        # 9. Upsample x to match low_level_feat size using F.interpolate
        # 10. low_level_feat = self.low_level_conv(low_level_feat)
        # 11. x = torch.cat([x, low_level_feat], dim=1) - Concatenate along channel dimension
        # 12. x = self.decoder(x)

        # CLASSIFICATION:
        # 13. x = self.classifier(x)
        # 14. Upsample x to original input size using F.interpolate
        # 15. Return x

        pass


# ================== Part 3: Mini-SAM Architecture ==================


class MiniSAM(nn.Module):
    """Simplified SAM architecture trainable from scratch (~5M parameters)"""

    def __init__(self, n_classes=21, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # TODO Task 3.1: Create lightweight image encoder
        # 1. Load models.mobilenet_v3_small(pretrained=True)
        # 2. Extract features: nn.Sequential(*list(backbone.features))
        # 3. Create projection: nn.Conv2d(576, embed_dim, 1) - MobileNetV3-Small outputs 576 channels

        # TODO Task 3.2: Create prompt encoders
        # Point type embedding:
        # 1. nn.Embedding(2, embed_dim) - 2 types: foreground (1) and background (0)

        # Point position embedding:
        # 2. Create nn.Sequential with:
        #    - nn.Linear(2, 128) - Input is (x, y) coordinates
        #    - nn.ReLU()
        #    - nn.Linear(128, embed_dim)

        # Box embedding:
        # 3. Create nn.Sequential with:
        #    - nn.Linear(4, 128) - Input is (x1, y1, x2, y2)
        #    - nn.ReLU()
        #    - nn.Linear(128, embed_dim)

        # TODO Task 3.3: Create decoder
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(embed_dim * 2, 256, 3, padding=1) - Image + prompt features
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU()
        #    - nn.Conv2d(256, 128, 3, padding=1)
        #    - nn.BatchNorm2d(128)
        #    - nn.ReLU()
        #    - nn.Conv2d(128, 64, 3, padding=1)
        #    - nn.BatchNorm2d(64)
        #    - nn.ReLU()

        # TODO Task 3.3: Create output heads
        # Mask head:
        # 1. nn.Conv2d(64, n_classes, 1)

        # IoU prediction head:
        # 2. Create nn.Sequential with:
        #    - nn.AdaptiveAvgPool2d(1)
        #    - nn.Flatten()
        #    - nn.Linear(64, 1)
        #    - nn.Sigmoid()

        # Upsampling:
        # 3. nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        pass

    def encode_image(self, x):
        """Extract image features"""
        # TODO Task 3.4: Encode image
        # 1. features = self.image_encoder(x) - Output: B x 576 x H/8 x W/8
        # 2. features = self.img_proj(features) - Output: B x embed_dim x H/8 x W/8
        # 3. Return features

        pass

    def encode_prompts(self, points=None, point_labels=None, boxes=None, img_size=None):
        """
        Encode point and/or box prompts
        Args:
            points: (B, N, 2) normalized [0,1] coordinates
            point_labels: (B, N) with 0=bg, 1=fg
            boxes: (B, 4) normalized [0,1] as [x1, y1, x2, y2]
            img_size: (H, W)
        Returns:
            prompt_features: (B, embed_dim, H/8, W/8)
        """
        # TODO Task 3.4: Encode prompts
        # 1. Get batch size B from points or boxes
        # 2. Get H, W from img_size
        # 3. Create empty list: prompt_features = []

        # IF POINTS PROVIDED:
        # 4. pos_enc = self.point_pos_embed(points) - Shape: B x N x embed_dim
        # 5. type_enc = self.point_type_embed(point_labels) - Shape: B x N x embed_dim
        # 6. point_enc = pos_enc + type_enc
        # 7. point_enc = point_enc.mean(dim=1, keepdim=True) - Average over N points
        # 8. Append point_enc to prompt_features

        # IF BOXES PROVIDED:
        # 9. box_enc = self.box_embed(boxes).unsqueeze(1) - Shape: B x 1 x embed_dim
        # 10. Append box_enc to prompt_features

        # COMBINE PROMPTS:
        # 11. If prompt_features not empty:
        #     - Concatenate along dim=1 and take mean: prompt_enc = torch.cat(...).mean(dim=1)
        #     - Reshape to (B, embed_dim, 1, 1)
        #     - Expand to (B, embed_dim, H//8, W//8)
        # 12. Else: create zero tensor of shape (B, embed_dim, H//8, W//8)
        # 13. Return prompt_enc

        pass

    def forward(self, images, points=None, point_labels=None, boxes=None):
        """
        Forward pass
        Args:
            images: B x 3 x H x W
            points: B x N x 2
            point_labels: B x N
            boxes: B x 4
        Returns:
            mask_logits: B x n_classes x H x W
            iou_pred: B x 1
        """
        # TODO Task 3.4: Implement forward pass
        # 1. Get B, C, H, W = images.shape

        # ENCODE IMAGE:
        # 2. img_features = self.encode_image(images)

        # ENCODE PROMPTS:
        # 3. prompt_features = self.encode_prompts(points, point_labels, boxes, img_size=(H, W))

        # FUSE FEATURES:
        # 4. fused = torch.cat([img_features, prompt_features], dim=1)

        # DECODE:
        # 5. decoded = self.decoder(fused)

        # OUTPUT MASK:
        # 6. mask_logits = self.mask_head(decoded)
        # 7. mask_logits = self.upsample(mask_logits)

        # PREDICT IoU:
        # 8. iou_pred = self.iou_head(decoded)

        # 9. Return mask_logits, iou_pred

        pass


def sample_points_from_mask(masks, n_points=5):
    """
    Sample points from ground truth masks (simulates user clicks)
    Args:
        masks: (B, H, W) ground truth class indices
        n_points: number of points to sample
    Returns:
        points: (B, n_points, 2) normalized coordinates
        labels: (B, n_points) with 0=bg, 1=fg
    """
    # TODO Task 3.5: Implement point sampling
    # 1. Get B, H, W = masks.shape
    # 2. Create empty lists: points_list = [], labels_list = []

    # FOR EACH IMAGE IN BATCH:
    # 3. For b in range(B):
    #    - Get mask = masks[b]

    #    SAMPLE FOREGROUND POINTS (50%):
    #    4. fg_indices = torch.nonzero(mask > 0) - Find all foreground pixels
    #    5. If fg_indices not empty:
    #       - Randomly sample n_points//2 indices
    #       - Normalize to [0,1]: divide by [H, W]
    #       - Create labels as ones
    #    6. Else: create empty tensors

    #    SAMPLE BACKGROUND POINTS (50%):
    #    7. bg_indices = torch.nonzero(mask == 0) - Find all background pixels
    #    8. If bg_indices not empty:
    #       - Randomly sample n_points//2 indices
    #       - Normalize to [0,1]: divide by [H, W]
    #       - Create labels as zeros
    #    9. Else: create empty tensors

    #    COMBINE AND PAD:
    #    10. Concatenate fg_points and bg_points
    #    11. Concatenate fg_labels and bg_labels
    #    12. If total points < n_points: pad with zeros
    #    13. Append to lists

    # 14. Stack lists and return torch.stack(points_list), torch.stack(labels_list)

    pass


# ================== Loss Functions ==================


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        # TODO Task 4.1: Implement Dice loss
        # 1. Apply softmax to pred: pred = torch.softmax(pred, dim=1)
        # 2. Convert target to one-hot: target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        # 3. Permute target_one_hot to (B, C, H, W) and convert to float
        # 4. Flatten spatial dimensions: pred_flat = pred.view(B, C, -1)
        # 5. Flatten target: target_flat = target_one_hot.view(B, C, -1)
        # 6. Compute intersection: (pred_flat * target_flat).sum(dim=2)
        # 7. Compute dice = (2 * intersection + smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + smooth)
        # 8. Return 1 - dice.mean()

        pass


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        # TODO Task 4.2: Implement Focal loss
        # 1. Compute cross-entropy: ce_loss = F.cross_entropy(pred, target, reduction='none')
        # 2. Compute pt = torch.exp(-ce_loss) - Probability of correct class
        # 3. Apply focal term: focal_loss = alpha * (1 - pt)^gamma * ce_loss
        # 4. Return focal_loss.mean()

        pass


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + Focal"""

    def __init__(self, weights={"ce": 0.3, "dice": 0.5, "focal": 0.2}):
        super().__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        # TODO Task 4.3: Compute weighted combination
        # 1. Compute: ce = self.ce_loss(pred, target)
        # 2. Compute: dice = self.dice_loss(pred, target)
        # 3. Compute: focal = self.focal_loss(pred, target)
        # 4. Return: weights['ce'] * ce + weights['dice'] * dice + weights['focal'] * focal

        pass


# ================== Evaluation Metrics ==================


def calculate_miou(pred, target, num_classes):
    """
    Calculate mean Intersection over Union
    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
        num_classes: int
    Returns:
        miou: float
        class_iou: numpy array (num_classes,)
    """
    # TODO Task 4.4: Implement mIoU
    # 1. Create empty list: ious = []
    # 2. Convert pred and target to numpy
    # 3. For each class c in range(num_classes):
    #    - pred_mask = (pred == c)
    #    - target_mask = (target == c)
    #    - intersection = logical_and(pred_mask, target_mask).sum()
    #    - union = logical_or(pred_mask, target_mask).sum()
    #    - If union == 0: iou = nan (class not present)
    #    - Else: iou = intersection / union
    #    - Append iou to ious
    # 4. Convert ious to numpy array
    # 5. Compute miou = nanmean(ious) - Ignores NaN values
    # 6. Return miou, ious

    pass


def calculate_pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    # TODO Task 4.5: Implement pixel accuracy
    # 1. correct = (pred == target).sum()
    # 2. total = pred.numel()
    # 3. Return (correct / total).item()

    pass


def compute_batch_iou(pred, target):
    """
    Compute IoU for each image in batch
    Args:
        pred: (B, H, W)
        target: (B, H, W)
    Returns:
        iou: (B,) tensor
    """
    # TODO Task 4.6: Implement batch IoU
    # 1. Get B = pred.shape[0]
    # 2. Create empty list: ious = []
    # 3. For each b in range(B):
    #    - intersection = ((pred[b] == target[b]) & (target[b] > 0)).float().sum()
    #    - union = ((pred[b] > 0) | (target[b] > 0)).float().sum()
    #    - iou = intersection / (union + 1e-6)
    #    - Append iou
    # 4. Return torch.stack(ious)

    pass


# ================== Training Functions ==================


def train_epoch_fcn(model, dataloader, optimizer, criterion, device):
    """Train FCN/DeepLab for one epoch"""
    model.train()
    total_loss = 0
    total_miou = 0

    # TODO Task 5.1: Implement training loop
    # 1. For images, masks in dataloader:
    # 2. Move to device: images, masks = images.to(device), masks.to(device)
    # 3. Zero gradients: optimizer.zero_grad()
    # 4. Forward pass: outputs = model(images)
    # 5. Compute loss: loss = criterion(outputs, masks)
    # 6. Backward: loss.backward()
    # 7. Update weights: optimizer.step()
    # 8. Calculate metrics: pred = outputs.argmax(dim=1), then miou = calculate_miou(pred, masks, num_classes)
    # 9. Accumulate: total_loss += loss.item(), total_miou += miou
    # 10. Return averages: total_loss / len(dataloader), total_miou / len(dataloader)

    pass


def train_epoch_minisam(model, dataloader, optimizer, criterion, device, n_points=5):
    """Train Mini-SAM for one epoch with simulated prompts"""
    model.train()
    total_loss = 0
    total_miou = 0

    # TODO Task 5.2: Implement Mini-SAM training
    # 1. For images, masks in dataloader:
    # 2. Move to device
    # 3. Sample points from masks: points, point_labels = sample_points_from_mask(masks, n_points)
    # 4. Zero gradients
    # 5. Forward pass: mask_logits, iou_pred = model(images, points, point_labels)
    # 6. Compute losses:
    #    - ce_loss = F.cross_entropy(mask_logits, masks)
    #    - dice_loss = DiceLoss()(mask_logits, masks)
    #    - pred_masks = mask_logits.argmax(dim=1)
    #    - true_iou = compute_batch_iou(pred_masks, masks)
    #    - iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)
    # 7. Combined loss: loss = ce_loss + dice_loss + 0.1 * iou_loss
    # 8. Backward and update
    # 9. Calculate metrics
    # 10. Return averages

    pass


def validate(model, dataloader, device, num_classes=21, is_minisam=False):
    """Validate the model"""
    model.eval()
    total_miou = 0
    total_pa = 0

    # TODO Task 5.3: Implement validation
    # 1. Use torch.no_grad() context
    # 2. For images, masks in dataloader:
    # 3. Move to device
    # 4. If is_minisam:
    #    - Sample points
    #    - outputs, _ = model(images, points, point_labels)
    #    Else:
    #    - outputs = model(images)
    # 5. Get predictions: pred = outputs.argmax(dim=1)
    # 6. Calculate metrics
    # 7. Accumulate
    # 8. Return averages

    pass


def visualize_predictions(
    model,
    dataloader,
    device,
    num_samples=4,
    is_minisam=False,
    save_path="predictions.png",
):
    """Visualize model predictions"""
    model.eval()

    # TODO Task 5.4: Implement visualization
    # 1. Get one batch: images_batch, masks_batch = next(iter(dataloader))
    # 2. Take first num_samples and move to device
    # 3. Generate predictions (with or without prompts based on is_minisam)
    # 4. Create figure with subplots: (num_samples, 3)
    # 5. For each sample, plot:
    #    - Column 0: Input image
    #    - Column 1: Ground truth mask
    #    - Column 2: Predicted mask
    # 6. Save figure

    pass


# ================== Main Training Script ==================


def main():
    """Main training pipeline"""

    config = {
        "model": "fcn8s",  # Options: 'fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam'
        "n_classes": 21,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 30,
        "device": device,
    }

    print(f"Training {config['model']} for {config['epochs']} epochs")

    # TODO Task 6.1: Setup data loaders
    # 1. Create dataset class or use existing PASCAL VOC dataset
    # 2. Create train_loader and val_loader with appropriate transforms

    # TODO Task 6.2: Create model
    # 1. Initialize model based on config['model']
    # 2. Move model to device

    # TODO Task 6.3: Setup optimizer and loss
    # 1. optimizer = torch.optim.AdamW(model.parameters(), lr=..., weight_decay=1e-4)
    # 2. scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
    # 3. criterion = CombinedLoss()

    # TODO Task 6.4: Training loop
    # 1. Initialize best_miou = 0
    # 2. For each epoch:
    #    - Train: train_loss, train_miou = train_epoch_*(...) based on model type
    #    - Validate: val_miou, val_pa = validate(...)
    #    - Update scheduler: scheduler.step(val_miou)
    #    - Print metrics
    #    - If val_miou > best_miou: save model

    # TODO Task 6.5: Plot training curves
    # 1. Plot training loss over epochs
    # 2. Plot validation mIoU over epochs
    # 3. Save plots

    pass


def compare_models():
    """Compare all implemented models"""
    # TODO Task 7.1: Compare FCN-32s, FCN-16s, FCN-8s, DeepLabV3+, and Mini-SAM
    # 1. Create results dictionary
    # 2. For each model:
    #    - Load trained checkpoint
    #    - Evaluate on test set
    #    - Record mIoU, number of parameters, inference time
    # 3. Create comparison table (use pandas or print nicely)
    # 4. Generate side-by-side qualitative comparisons

    pass


def interactive_minisam_demo():
    """Interactive Mini-SAM demo with point/box prompting"""
    # TODO Task 7.2: Create interactive demo
    # 1. Load trained Mini-SAM model
    # 2. Load test image
    # 3. Display image and get user clicks (or use pre-defined points)
    # 4. Run model with point prompts
    # 5. Display segmentation result
    # 6. Allow user to add correction points
    # 7. Re-run and display refined result

    pass


if __name__ == "__main__":
    print("Lab 5: Advanced Image Segmentation")
    print("=" * 60)
    print("\nComplete all TODOs in the order they appear!")
    print("\nRecommended implementation order:")
    print("1. Part 1: FCN-32s (simplest, no skip connections)")
    print("2. Part 1: FCN-16s (add one skip connection)")
    print("3. Part 1: FCN-8s (add two skip connections)")
    print("4. Part 2: ASPP module")
    print("5. Part 2: DeepLabV3+")
    print("6. Part 3: Mini-SAM (most challenging!)")
    print("7. Part 4: Loss functions")
    print("8. Part 5: Training and evaluation")
    print("\n" + "=" * 60)

    # Uncomment to run training
    # main()

    # Uncomment to run comparisons
    # compare_models()

    # Uncomment to run interactive demo
    # interactive_minisam_demo()

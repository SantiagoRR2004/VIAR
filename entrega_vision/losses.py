from torch import nn
import torch

# ================== Part 3: Loss Functions ==================


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Task 3.1: Implement Dice loss
        # 1. Apply sigmoid to predictions
        pred = torch.sigmoid(pred)

        # 2. Flatten both pred and target
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 3. Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # 4. Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 5. Return 1 - dice
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target):
        # Task 3.2: Implement Focal loss
        # 1. Apply sigmoid and compute BCE
        pred = torch.sigmoid(pred)
        bce = self.ce(pred, target)

        # 2. Calculate p_t (probability of correct class)
        p_t = torch.exp(-bce)

        # 3. Apply focal term: (1-p_t)^gamma
        focalTerm = (1 - p_t) ** self.gamma

        # 4. Apply alpha weighting
        loss = self.alpha * focalTerm * bce
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function"""

    def __init__(self, weights={"ce": 0.5, "dice": 0.5, "focal": 0.0}):
        super().__init__()
        # Task 3.3: Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

    def forward(self, pred, target):
        # Task 3.3: Compute weighted combination of losses
        total_loss = 0

        # Add each loss component with its weight
        total_loss += self.weights.get("ce", 0) * self.ce_loss(pred, target)
        total_loss += self.weights.get("dice", 0) * self.dice_loss(pred, target)
        total_loss += self.weights.get("focal", 0) * self.focal_loss(pred, target)

        return total_loss

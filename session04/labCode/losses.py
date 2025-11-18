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


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: softmax (B, C, H, W)
        # targets: one hot (B, C, H, W)
        preds = torch.softmax(preds, dim=1)
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


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


class MultiClassFocalLoss(nn.Module):
    """Focal loss for multiclass segmentation"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target):
        # pred: logits (B, C, H, W)
        # target: class indices (B, H, W)
        if target.dim() == 4:
            target = target.argmax(dim=1)  # Convert one-hot to class indices

        # 1) Cross entropy (per pixel)
        ce_loss = self.ce(pred, target)  # shape (B,H,W)

        # 2) Softmax probabilities
        probs = torch.softmax(pred, dim=1)  # (B,C,H,W)

        # 3) Probabilidad de la clase correcta
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,H,W)

        # 4) (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # 5) Alpha weight
        loss = self.alpha * focal_weight * ce_loss

        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function"""

    def __init__(self, weights={"ce": 0.1, "dice": 0.3, "focal": 0.60}):
        super().__init__()
        # Task 3.3: Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss()
        self.focal_loss = MultiClassFocalLoss()
        self.weights = weights

    def forward(self, pred, target):
        # Task 3.3: Compute weighted combination of losses
        total_loss = 0

        # Add each loss component with its weight
        total_loss += self.weights.get("ce", 0) * self.ce_loss(pred, target)
        total_loss += self.weights.get("dice", 0) * self.dice_loss(pred, target)
        total_loss += self.weights.get("focal", 0) * self.focal_loss(pred, target)

        return total_loss

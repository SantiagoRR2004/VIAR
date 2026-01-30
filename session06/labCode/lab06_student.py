# ============================================================================
# Lab 6: Object Detection - Student Template
# Artificial Vision (VIAR25/26) - UVigo
#
# INSTRUCTIONS:
# 1. Complete the TODOs in order
# 2. Test each component before moving to the next
# 3. Use the provided test functions to verify your implementation
# 4. Refer to the lecture notes for mathematical details
# ============================================================================

from datasets import COCODetectionDataset, get_transforms
from utils.nms import compute_iou, nms
from models.yolo import YOLODetector
from models.fcos import FCOSDetector
from config import config
import torch

# ============================================================================
# TEST FUNCTIONS (Use these to verify your implementations)
# ============================================================================


def test_dataset():
    """Test dataset implementation"""
    print("Testing dataset...")

    # Create dataset
    dataset = COCODetectionDataset(
        config.train_images,
        config.train_ann,
        transform=get_transforms(is_train=True),
        is_train=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Get one sample
    image, boxes, labels, img_id = dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image ID: {img_id}")

    # Verify boxes are in correct format
    assert image.shape == (3, 448, 448), "Image shape incorrect"
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "Boxes shape incorrect"
    assert labels.ndim == 1, "Labels shape incorrect"
    assert torch.all((boxes >= 0) & (boxes <= 1)), "Boxes not normalized"

    print("Dataset test passed!")


def test_iou():
    """Test IoU computation"""
    print("Testing IoU...")

    # Test case 1: Perfect overlap
    box1 = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    box2 = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    iou = compute_iou(box1, box2)
    assert torch.isclose(
        iou, torch.tensor(1.0)
    ), f"Perfect overlap should be 1.0, got {iou}"

    # Test case 2: No overlap
    box1 = torch.tensor([[0.0, 0.0, 0.3, 0.3]])
    box2 = torch.tensor([[0.7, 0.7, 1.0, 1.0]])
    iou = compute_iou(box1, box2)
    assert torch.isclose(iou, torch.tensor(0.0)), f"No overlap should be 0.0, got {iou}"

    # Test case 3: Partial overlap
    box1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    box2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    iou = compute_iou(box1, box2)
    expected = 0.25 * 0.25 / (0.5 * 0.5 + 0.5 * 0.5 - 0.25 * 0.25)
    assert torch.isclose(
        iou, torch.tensor(expected), atol=1e-4
    ), f"IoU incorrect: {iou} vs {expected}"

    print("IoU test passed!")


def test_nms():
    """Test NMS implementation"""
    print("Testing NMS...")

    # Create overlapping boxes with different scores
    boxes = torch.tensor(
        [
            [0.1, 0.1, 0.4, 0.4],  # High score
            [0.15, 0.15, 0.45, 0.45],  # Overlaps with first, lower score
            [0.6, 0.6, 0.9, 0.9],  # Different location, medium score
        ]
    )
    scores = torch.tensor([0.9, 0.7, 0.8])

    keep = nms(boxes, scores, iou_threshold=0.5)

    # Should keep first and third box
    assert len(keep) == 2, f"Should keep 2 boxes, kept {len(keep)}"
    assert 0 in keep, "Should keep highest scoring box"
    assert 2 in keep, "Should keep non-overlapping box"

    print("NMS test passed!")


def test_yolo_model():
    """Test YOLO model"""
    print("Testing YOLO model...")

    model = YOLODetector(num_classes=80, grid_size=7, num_boxes=2)

    # Test forward pass
    x = torch.randn(2, 3, 448, 448)
    predictions = model(x)

    assert predictions.shape == (
        2,
        7,
        7,
        2,
        85,
    ), f"Output shape incorrect: {predictions.shape}"

    # Check activations
    assert torch.all(
        (predictions[..., 0:2] >= 0) & (predictions[..., 0:2] <= 1)
    ), "x, y not in [0,1]"
    assert torch.all(
        (predictions[..., 4] >= 0) & (predictions[..., 4] <= 1)
    ), "confidence not in [0,1]"
    assert torch.allclose(
        predictions[..., 5:].sum(dim=-1), torch.ones(2, 7, 7, 2)
    ), "class probs don't sum to 1"

    print("YOLO model test passed!")


def test_fcos_model():
    """Test FCOS model"""
    print("Testing FCOS model...")

    model = FCOSDetector(num_classes=80)

    # Test forward pass
    x = torch.randn(2, 3, 448, 448)
    cls_logits, reg_preds, centerness = model(x)

    assert len(cls_logits) == 3, "Should have 3 FPN levels"
    assert len(reg_preds) == 3, "Should have 3 FPN levels"
    assert len(centerness) == 3, "Should have 3 FPN levels"

    for cls, reg, cent in zip(cls_logits, reg_preds, centerness):
        assert cls.size(1) == 80, "Classification channels incorrect"
        assert reg.size(1) == 4, "Regression channels incorrect"
        assert cent.size(1) == 1, "Centerness channels incorrect"

    print("FCOS model test passed!")


def run_all_tests():
    """Run all test functions"""
    print("=" * 50)
    print("Running all tests...")
    print("=" * 50)

    test_dataset()
    test_iou()
    test_nms()
    test_yolo_model()
    test_fcos_model()

    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    # Uncomment to run tests
    run_all_tests()

    # Uncomment to start training
    # main()

    print("Lab 6 Template Ready!")
    print("\nNext steps:")
    print("1. Complete the TODOs in order")
    print("2. Run tests after each major component: run_all_tests()")
    print("3. Start training: main()")
    print("\nGood luck!")

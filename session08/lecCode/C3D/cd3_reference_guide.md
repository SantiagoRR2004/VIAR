# C3D Demo & Fine-Tuning Quick Reference Card

## 📥 Download Pretrained Weights

```bash
# Download Sports-1M weights (DavideA)
wget -O c3d_sports1m.pickle \
  https://github.com/DavideA/c3d-pytorch/releases/download/v0.1/c3d.pickle
```

---

## 🎬 Demo Usage

### Sports-1M Mode (487 classes - Meaningful predictions)
```bash
python c3d_demo_pretrained.py \
  --video your_video.mp4 \
  --weights c3d_sports1m.pickle \
  --labels sports1M_labels.txt \
  --num_classes 487 \
  --device cuda
```

**Expected**: Correct predictions (e.g., "golf" 65%, "tennis" 8%)

### UCF-101 Mode Before Fine-Tuning (101 classes - Random predictions)
```bash
python c3d_demo_pretrained.py \
  --video your_video.mp4 \
  --weights c3d_sports1m.pickle \
  --labels classInd.txt \
  --num_classes 101 \
  --device cuda
```

**Expected**: Random predictions (~1% each) - THIS IS NORMAL! Fine-tune to fix.

### UCF-101 Mode After Fine-Tuning (101 classes - Good predictions)
```bash
python c3d_demo_pretrained.py \
  --video your_video.mp4 \
  --weights checkpoints_c3d/best.pth \  # your fine-tuned checkpoint
  --labels classInd.txt \
  --num_classes 101 \
  --device cuda
```

**Expected**: Correct predictions (e.g., "GolfSwing" 92%)

---

## 🎓 Fine-Tuning on UCF-101

### Quick Start (fc policy - 6-8 hours)
```bash
python finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /path/to/UCF-101 \
  --weights c3d_sports1m.pickle \
  --split 1 \
  --freeze fc \
  --epochs 8 \
  --batch_size 8 \
  --segments 3 \
  --device cuda \
  --checkpoint_dir ./checkpoints_c3d
```

**Result**: 75-80% accuracy, fast training

### Recommended (conv5 policy - 10-12 hours)
```bash
python finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /path/to/UCF-101 \
  --weights c3d_sports1m.pickle \
  --split 1 \
  --freeze conv5 \
  --epochs 8 \
  --batch_size 8 \
  --segments 3 \
  --amp \  # Enable mixed precision
  --device cuda \
  --checkpoint_dir ./checkpoints_c3d
```

**Result**: 80-85% accuracy, good trade-off

### Maximum Accuracy (none policy - 15-20 hours)
```bash
python finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /path/to/UCF-101 \
  --weights c3d_sports1m.pickle \
  --split 1 \
  --freeze none \
  --epochs 10 \
  --batch_size 6 \  # Smaller batch due to more memory
  --segments 3 \
  --amp \
  --device cuda \
  --checkpoint_dir ./checkpoints_c3d
```

**Result**: 82-85% accuracy, slower training

---

## 🎯 Freeze Policy Comparison

| Policy | Trainable Layers | Params | Time | Accuracy | GPU RAM | Best For |
|--------|------------------|--------|------|----------|---------|----------|
| **fc** | fc6, fc7, fc8 | 45M (58%) | 6-8h | 75-80% | 6 GB | Quick experiments |
| **conv5** ⭐ | fc6-8 + conv5a/b | 60M (77%) | 10-12h | 80-85% | 8 GB | **Production** |
| **none** | All layers | 78M (100%) | 15-20h | 82-85% | 10 GB | Max accuracy |

---

## 🔧 Important Parameters

### Multi-Clip Sampling
- `--segments 3` (training): Sample 3 random clips per video
- `--segments 5` (validation): Sample 5 uniform clips per video
- Higher K = better accuracy (+3-5%) but slower

### Batch Size
- `--batch_size 8`: Standard (requires 8-10 GB GPU)
- `--batch_size 6`: If memory limited
- `--batch_size 12`: If you have large GPU (24+ GB)

### Mixed Precision
- `--amp`: Enable for 2× memory reduction and 1.5-2× speedup
- Recommended for all training!

### Learning Rate
- `--lr 1e-3`: Default (good for fc, conv5 policies)
- `--lr 5e-4`: Lower for none policy (training all layers)
- `--lr 5e-3`: Higher if convergence too slow

---

## 🐛 Common Issues & Solutions

### Issue: "Missing keys: ['fc8.weight', 'fc8.bias']"
**Status**: ✅ **EXPECTED** - This is normal!
**Why**: Sports-1M has 487 classes, UCF-101 has 101 classes. fc8 is re-initialized.
**Action**: Proceed normally. The pretrained features (conv1-5, fc6-7) are loaded correctly.

### Issue: "RuntimeError: size mismatch for fc6.weight"
**Problem**: Wrong architecture variant
**Solution**: The code auto-detects! But if you forced wrong arch:
- Check `--force_arch sports1m` or `--force_arch ours`
- Let auto-detection work (don't use --force_arch)

### Issue: Demo predictions are random/nonsense
**Before fine-tuning**: ✅ **EXPECTED** - fc8 is randomly initialized
**After fine-tuning**: ❌ Problem! Check:
1. Did fine-tuning complete? (check checkpoint files)
2. Are you loading the right checkpoint? (use best.pth not latest)
3. Is --num_classes correct? (101 for UCF-101)

### Issue: CUDA out of memory
**Solutions**:
1. Reduce batch size: `--batch_size 4`
2. Reduce segments: `--segments 2`
3. Enable AMP: `--amp`
4. Use smaller freeze policy: `--freeze fc` instead of `--freeze conv5`

### Issue: Training accuracy stuck at ~1%
**Problem**: Layers might not be trainable
**Check**:
```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: trainable')
```
Should see fc6, fc7, fc8 (and conv5a/b if using conv5 policy).

### Issue: Validation accuracy not improving
**Potential causes**:
1. Learning rate too high/low → Try `--lr 5e-4` or `--lr 5e-3`
2. Overfitting → Add `--wd 5e-4` (weight decay)
3. Not enough epochs → Try `--epochs 12`
4. Need more trainable layers → Switch to `--freeze conv5` or `--freeze none`

---

## 📊 Expected Results Timeline

### FC Policy (Quick)
- **Epoch 1**: ~40-50% accuracy
- **Epoch 4**: ~65-70% accuracy
- **Epoch 8**: ~75-80% accuracy (plateau)
- **Time**: 6-8 hours

### Conv5 Policy (Recommended)
- **Epoch 1**: ~45-55% accuracy
- **Epoch 4**: ~70-75% accuracy
- **Epoch 8**: ~80-83% accuracy
- **Epoch 12**: ~82-85% accuracy (plateau)
- **Time**: 10-12 hours

### None Policy (Maximum)
- **Epoch 1**: ~50-60% accuracy
- **Epoch 5**: ~75-80% accuracy
- **Epoch 10**: ~82-85% accuracy (plateau)
- **Time**: 15-20 hours

---

## 🎯 Decision Tree: Which Settings Should I Use?

### I want to...

**...test if everything works**
→ Use fc policy, 2 epochs, batch_size 8
→ Time: ~1.5 hours
→ Expected: ~60-65% accuracy

**...get reasonable results quickly**
→ Use fc policy, 8 epochs, batch_size 8, --amp
→ Time: 6-8 hours
→ Expected: 75-80% accuracy

**...get good production results** ⭐
→ Use conv5 policy, 8 epochs, batch_size 8, --amp
→ Time: 10-12 hours
→ Expected: 80-85% accuracy

**...maximize accuracy (research)**
→ Use none policy, 10-12 epochs, batch_size 6, --amp
→ Time: 15-20 hours
→ Expected: 82-85% accuracy

**...save GPU memory**
→ Use fc policy, batch_size 4, --segments 2, --amp
→ Works with 4-6 GB GPU
→ Time: 8-10 hours
→ Expected: 73-78% accuracy

---

## 📝 Typical Training Log

### Good Training (Converging)
```
Epoch 1/8
[Train] loss 2.543 | avg_acc 42.3% | eta 65.2 min
[Val]   avg_acc 45.8%
Saved best checkpoint: 45.8%

Epoch 4/8
[Train] loss 0.982 | avg_acc 72.1% | eta 32.1 min
[Val]   avg_acc 74.3%
Saved best checkpoint: 74.3%

Epoch 8/8
[Train] loss 0.654 | avg_acc 81.2% | eta 0.0 min
[Val]   avg_acc 79.8%
Best accuracy: 79.8%
```

### Problem: Not Converging
```
Epoch 1/8
[Train] loss 4.612 | avg_acc 1.2%    # ← BAD: Stuck at random
[Val]   avg_acc 1.1%

Epoch 4/8
[Train] loss 4.609 | avg_acc 1.3%    # ← BAD: No improvement
[Val]   avg_acc 1.2%
```

**Diagnosis**: Layers not trainable or learning rate too low
**Fix**: Check freeze policy and increase lr to 5e-3

---

## 🔗 Useful Links

- **DavideA C3D PyTorch**: https://github.com/DavideA/c3d-pytorch
- **Original C3D Paper**: Tran et al. "Learning Spatiotemporal Features with 3D Convolutional Networks" (ICCV 2015)
- **Sports-1M Dataset**: Karpathy et al. "Large-scale Video Classification with Convolutional Neural Networks" (CVPR 2014)
- **UCF-101 Dataset**: http://www.crcv.ucf.edu/data/UCF101.php

---

## 💡 Pro Tips

1. **Always enable --amp**: 2× speedup, same accuracy
2. **Use conv5 policy**: Best accuracy/time trade-off
3. **Monitor both train and val accuracy**: If train >> val, you're overfitting
4. **Save checkpoints**: Use --checkpoint_dir to avoid losing progress
5. **Test before fine-tuning**: Run Sports-1M demo to verify weights loaded correctly
6. **Start small**: Test with 2 epochs first to ensure pipeline works
7. **Use tmux/screen**: Training takes hours, don't let terminal disconnect kill it!

---

## 🚀 Complete Workflow Example

```bash
# 1. Download weights (one-time, 5 minutes)
wget -O c3d_sports1m.pickle \
  https://github.com/DavideA/c3d-pytorch/releases/download/v0.1/c3d.pickle

# 2. Test Sports-1M prediction (verify weights work, 1 minute)
python c3d_demo_pretrained.py \
  --video ../dataset/golf.mp4 \
  --weights c3d_sports1m.pickle \
  --labels sports1M_labels.txt \
  --num_classes 487
# Should see: "golf" with high confidence

# 3. Test UCF-101 prediction before fine-tuning (verify fc8 randomness, 1 minute)
python c3d_demo_pretrained.py \
  --video ../dataset/golf.mp4 \
  --weights c3d_sports1m.pickle \
  --labels classInd.txt \
  --num_classes 101
# Should see: Random predictions ~1% each

# 4. Fine-tune on UCF-101 (10-12 hours)
python finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /path/to/UCF-101 \
  --weights c3d_sports1m.pickle \
  --split 1 \
  --freeze conv5 \
  --epochs 8 \
  --batch_size 8 \
  --segments 3 \
  --amp \
  --device cuda \
  --checkpoint_dir ./checkpoints_c3d \
  --log_interval 25
# Wait for: "Best accuracy: 80.2%" or similar

# 5. Test UCF-101 prediction after fine-tuning (verify it works, 1 minute)
python c3d_demo_pretrained.py \
  --video ../dataset/golf.mp4 \
  --weights checkpoints_c3d/best.pth \
  --labels classInd.txt \
  --num_classes 101
# Should see: "GolfSwing" with high confidence (85-95%)

# 6. Test on multiple videos to verify generalization
python c3d_demo_pretrained.py \
  --video ../dataset/basketball.mp4 \
  --weights checkpoints_c3d/best.pth \
  --labels classInd.txt \
  --num_classes 101
# Should see: "BasketballDunk" or similar with high confidence
```

**Total time**: ~10-13 hours (mostly training)
**Expected final accuracy**: 80-85% on UCF-101

---

**Remember**: Random predictions before fine-tuning are EXPECTED and NORMAL! The magic happens during fine-tuning. 🚀










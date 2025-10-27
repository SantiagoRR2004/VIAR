import argparse, os
from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# ----------------------
# Two C3D variants
# ----------------------
class C3D_Ours_4608(nn.Module):
    """C3D variant with pool5 no padding => fc6 in_features=4608"""
    def __init__(self, num_classes=101, dropout=0.5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1);           
        self.pool1 = nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv2 = nn.Conv3d(64,128,3, padding=1);            
        self.pool2 = nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv3a= nn.Conv3d(128,256,3,padding=1);            
        self.conv3b= nn.Conv3d(256,256,3,padding=1); 
        self.pool3=nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv4a= nn.Conv3d(256,512,3,padding=1);            
        self.conv4b= nn.Conv3d(512,512,3,padding=1); 
        self.pool4=nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv5a= nn.Conv3d(512,512,3,padding=1);            
        self.conv5b= nn.Conv3d(512,512,3,padding=1); 
        self.pool5=nn.MaxPool3d((2,2,2),(2,2,2))
        self.fc6 = nn.Linear(512*1*3*3, 4096)  # 4608
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool1(x)
        x = self.relu(self.conv2(x)); x = self.pool2(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool3(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x)); x = self.pool4(x)
        x = self.relu(self.conv5a(x)); x = self.relu(self.conv5b(x)); x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc6(x)))
        x = self.drop(F.relu(self.fc7(x)))
        return self.fc8(x)


class C3D_Sports1M_8192(nn.Module):
    """Canonical Sports-1M/UCF101 C3D (pool5 padding=(0,1,1) => fc6 in_features=8192)"""
    def __init__(self, num_classes=101, dropout=0.5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1) 
        self.pool1 = nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv2 = nn.Conv3d(64,128,3, padding=1)
        self.pool2 = nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv3a= nn.Conv3d(128,256,3,padding=1)
        self.conv3b= nn.Conv3d(256,256,3,padding=1) 
        self.pool3=nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv4a= nn.Conv3d(256,512,3,padding=1) 
        self.conv4b= nn.Conv3d(512,512,3,padding=1) 
        self.pool4=nn.MaxPool3d((2,2,2),(2,2,2))
        self.conv5a= nn.Conv3d(512,512,3,padding=1) 
        self.conv5b= nn.Conv3d(512,512,3,padding=1) 
        self.pool5=nn.MaxPool3d((2,2,2),(2,2,2), padding=(0,1,1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool1(x)
        x = self.relu(self.conv2(x)); x = self.pool2(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool3(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x)); x = self.pool4(x)
        x = self.relu(self.conv5a(x)); x = self.relu(self.conv5b(x)); x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc6(x)))
        x = self.drop(F.relu(self.fc7(x)))
        return self.fc8(x)


# ----------------------
# Video utils (112x112, 16f)
# ----------------------
def read_video_rgb(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    try:
        while True:
            ret, f = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) < 16:
        raise RuntimeError("Video too short (<16 frames).")
    return frames

def resize_short_side(img: np.ndarray, short=128) -> np.ndarray:
    h,w = img.shape[:2]
    s = short / min(h,w)
    return cv2.resize(img, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)

def center_crop(img: np.ndarray, size=112) -> np.ndarray:
    h,w = img.shape[:2]
    y=(h-size)//2; x=(w-size)//2
    return img[y:y+size, x:x+size, :]

def preprocess(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)/255.0
    x = (x - 0.5)/0.5
    return x

def make_clips(frames: list[np.ndarray], clip_len=16, stride=16, max_clips=None) -> list[np.ndarray]:
    T=len(frames); clips=[]
    for s in range(0, T-clip_len+1, stride):
        buf=[]
        for t in range(s, s+clip_len):
            f = preprocess(center_crop(resize_short_side(frames[t],128),112))
            buf.append(f)
        clips.append(np.stack(buf,0))  # [T,112,112,3]
        if max_clips and len(clips)>=max_clips: break
    if not clips:
        buf=[]
        for t in range(T-clip_len, T):
            f = preprocess(center_crop(resize_short_side(frames[t],128),112))
            buf.append(f)
        clips=[np.stack(buf,0)]
    return clips

def clips_to_tensor(clips: list[np.ndarray]) -> torch.Tensor:
    # [N,T,H,W,3] -> [N,3,T,H,W]
    xs=[]
    for c in clips:
        xs.append(torch.from_numpy(np.transpose(c,(3,0,1,2))))
    return torch.stack(xs,0)


# ----------------------
# Labels helper
# ----------------------
def load_ucf101_labels(classInd: str|None, n=101)->list[str]:
    if not classInd or not os.path.isfile(classInd): return [f"class_{i}" for i in range(n)]
    names=[None]*n
    with open(classInd) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            idx,name=line.split()
            names[int(idx)-1]=name
    return names

def load_labels_txt(path: str | None, n_default: int) -> list[str]:
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [f"class_{i}" for i in range(n_default)]


# ----------------------
# Checkpoint loading
# ----------------------
def strip_prefix_if_present(state, prefix="module."):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k,v in state.items() }

def guess_arch_from_ckpt(state):
    # Look at fc6.weight shape: [4096, 8192] => Sports1M; [4096, 4608] => Ours
    w = state.get("fc6.weight", None)
    if w is None:
        # try some common name variants (mmaction2 sometimes prefixes)
        for k in state.keys():
            if k.endswith("fc6.weight"):
                w = state[k]; break
    if w is None:  # default to Sports1M (most public weights)
        return "sports1m"
    return "sports1m" if w.shape[1]==8192 else "ours"

def load_c3d_weights(model: nn.Module, ckpt_path: str) -> tuple[list[str], list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_prefix_if_present(state, "module.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected





def load_c3d_pickle_into_model(model, ckpt_path, drop_fc8_if_mismatch: bool = True):
    """
    Safely load DavideA Sports-1M C3D weights (.pickle/.pth) into your model.
        https://github.com/DavideA/c3d-pytorch
    - Tries torch.load(..., weights_only=True) first (safer).
    - Falls back to legacy torch.load, then latin1 pickle for old files.
    - Strips 'module.' prefixes from DataParallel.
    - Optionally drops fc8.* if class count mismatches (e.g., 487 vs 101).
    Returns: (missing_keys, unexpected_keys)
    """
    # --- robust load ---
    ckpt = None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        # Older PyTorch without weights_only
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f, encoding="latin1")

    # --- get state dict ---
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    else:
        state = ckpt  # many files are already a raw state_dict

    # --- strip DataParallel prefix ---
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    # --- optionally drop fc8 if size mismatches ---
    if drop_fc8_if_mismatch and "fc8.weight" in state:
        want = model.fc8.weight.shape[0]
        have = state["fc8.weight"].shape[0]
        if want != have:
            state.pop("fc8.weight", None)
            state.pop("fc8.bias", None)

    # --- load with relaxed matching ---
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected



def preprocess_frame_sports1m(f: np.ndarray) -> np.ndarray:
    # resize to (H=112, W=200) then center-crop to 112x112, no mean/std
    f = cv2.resize(f, (200, 112), interpolation=cv2.INTER_AREA)
    x0 = (200 - 112) // 2  # 44
    f = f[:, x0:x0+112, :]    # shape (112,112,3)
    return f.astype(np.float32)

def make_clips_sports1m(frames: list[np.ndarray], clip_len=16, stride=16, max_clips=None) -> list[np.ndarray]:
    T=len(frames); clips=[]
    for s in range(0, T-clip_len+1, stride):
        buf=[preprocess_frame_sports1m(frames[t]) for t in range(s, s+clip_len)]
        clips.append(np.stack(buf,0))  # [T,112,112,3]
        if max_clips and len(clips)>=max_clips: break
    if not clips:
        buf=[preprocess_frame_sports1m(frames[t]) for t in range(T-clip_len, T)]
        clips=[np.stack(buf,0)]
    return clips



# ----------------------
# Main
# ----------------------
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=False, default=None, help="Path to C3D checkpoint (.pickle/.pth)")
    ap.add_argument("--labels", required=False, default=None, help="Sports-1M labels.txt (487) OR UCF101 classInd.txt (101)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--num_classes", type=int, default=101, help="487 for Sports-1M prediction, 101 for UCF101/fine-tune")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--force_arch", choices=["ours","sports1m"], default=None)
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu")

    # ---------- Load weights (if any) and guess arch BEFORE preprocessing ----------
    state = None
    arch = "ours"
    if args.weights:
        # robust load for .pickle/.pth (older pickles may need latin1)
        try:
            raw = torch.load(args.weights, map_location="cpu")
        except Exception:
            with open(args.weights, "rb") as f:
                raw = pickle.load(f, encoding="latin1")
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state = strip_prefix_if_present(state, "module.")
        arch = guess_arch_from_ckpt(state)  # "sports1m" or "ours"

    if args.force_arch:
        arch = args.force_arch

    # ---------- Read video and build clips using arch-appropriate preprocessing ----------
    frames = read_video_rgb(args.video)
    if arch == "sports1m":
        clips = make_clips_sports1m(frames, clip_len=16, stride=args.stride, max_clips=args.max_clips)
    else:
        clips = make_clips(frames, clip_len=16, stride=args.stride, max_clips=args.max_clips)
    x = clips_to_tensor(clips).to(dev)  # [N,3,16,112,112]

    # ---------- Build model ----------
    # If you're predicting Sports-1M classes, pass --num_classes 487.
    if arch == "sports1m":
        model = C3D_Sports1M_8192(num_classes=args.num_classes).to(dev)
    else:
        model = C3D_Ours_4608(num_classes=args.num_classes).to(dev)

    # ---------- Load checkpoint into model (drop fc8 if class-count mismatch) ----------
    if state is not None:
        missing, unexpected = load_c3d_pickle_into_model(model, args.weights, drop_fc8_if_mismatch=True)
        print(f"Loaded weights from: {args.weights}")
        if missing:    print("  missing:", missing)
        if unexpected: print("  unexpected:", unexpected)
    else:
        print("No weights provided -> random init (mechanics demo).")

    # ---------- Labels ----------
    if args.num_classes == 487:
        # Sports-1M: plain labels.txt (one name per line)
        labels = load_labels_txt(args.labels, n_default=487)
    elif args.num_classes == 101:
        # UCF101: classInd.txt (index name)
        labels = load_ucf101_labels(args.labels, n=101)
    else:
        labels = load_labels_txt(args.labels, n_default=args.num_classes)

    # ---------- Inference (avg over clips) ----------
    model.eval()
    with torch.no_grad():
        logits_sum = torch.zeros(1, args.num_classes, device=dev)
        for i in range(x.shape[0]):
            logits_sum += model(x[i:i+1])
        probs = torch.softmax(logits_sum / x.shape[0], dim=1)[0]
        vals, idxs = torch.topk(probs, k=5)

    print(f"\nVideo: {args.video} | clips: {x.shape[0]} | arch: {arch} | classes: {args.num_classes}")
    print("Top-5 (avg over clips):")
    for r,(p,i) in enumerate(zip(vals, idxs), 1):
        name = labels[i.item()] if 0 <= i.item() < len(labels) else f"class_{i.item()}"
        print(f"  {r}. {name:<30} {float(p)*100:5.2f}%")



if __name__ == "__main__":
    main()




"""
# python c3d_demo_pretrained.py \
#   --video ../dataset/vgolf2.mpg \
#   --weights /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/c3d_sports1m.pth \
#   --labels /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/ucfTrainTestlist/classInd.txt \
#   --device cuda

"""



"""
Sports-1M prediction (487 classes):

python c3d_demo_pretrained.py \
  --video ../dataset/vgolf2.mpg \
  --weights /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/c3d_sports1m.pickle \
  --labels ../dataset/sports1M_labels.txt \
  --device cuda \
  --num_classes 487 \
  --stride 8 --max_clips 8


"""

import cv2, numpy as np, os
os.makedirs("figs", exist_ok=True)

# --- Read grayscale ---
img_path = "figs/canny_input.jpg"  # <-- replace this
I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
assert I is not None, "Could not read input image"

# --- Step 1: Gaussian smoothing (σ ≈ 1.4 is classical) ---
sigma = 1.4
Is = cv2.GaussianBlur(I, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)

# --- Step 2: Gradients (Derivative-of-Gaussian ≈ Sobel after blur) ---
Ix = cv2.Sobel(Is, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(Is, cv2.CV_32F, 0, 1, ksize=3)

# --- Step 3: Magnitude & Direction ---
mag = np.hypot(Ix, Iy)
ang = (np.degrees(np.arctan2(Iy, Ix)) + 180.0) % 180.0  # 0..180

# --- Step 4: Non-Maximum Suppression (quantize to {0,45,90,135}) ---
def nms(mag, ang):
    H, W = mag.shape
    out = np.zeros_like(mag, dtype=np.float32)
    # Quantize direction
    q = np.zeros_like(ang, dtype=np.uint8)
    # bins: [-22.5,22.5)->0 ; [22.5,67.5)->45 ; [67.5,112.5)->90 ; [112.5,157.5)->135 ; wrap
    ang_q = ang.copy()
    q[(ang_q < 22.5) | (ang_q >= 157.5)] = 0
    q[(ang_q >= 22.5) & (ang_q < 67.5)] = 45
    q[(ang_q >= 67.5) & (ang_q < 112.5)] = 90
    q[(ang_q >= 112.5) & (ang_q < 157.5)] = 135

    for i in range(1, H-1):
        for j in range(1, W-1):
            m = mag[i, j]
            if q[i, j] == 0:
                n1, n2 = mag[i, j-1], mag[i, j+1]
            elif q[i, j] == 45:
                n1, n2 = mag[i-1, j+1], mag[i+1, j-1]
            elif q[i, j] == 90:
                n1, n2 = mag[i-1, j], mag[i+1, j]
            else:  # 135
                n1, n2 = mag[i-1, j-1], mag[i+1, j+1]
            if m >= n1 and m >= n2:
                out[i, j] = m
    return out

mag_nms = nms(mag, ang)

# --- Step 5: Double threshold ---
# Use fractions of max magnitude; tune per image/domain
T_low_frac, T_high_frac = 0.10, 0.25
T_low, T_high = T_low_frac*mag_nms.max(), T_high_frac*mag_nms.max()

strong = (mag_nms >= T_high)
weak   = (mag_nms >= T_low) & ~strong

# --- Step 6: Hysteresis (8-neighborhood) ---
E = np.zeros_like(strong, dtype=np.uint8)
visited = np.zeros_like(strong, dtype=np.uint8)
from collections import deque
Q = deque(list(zip(*np.nonzero(strong))))
for i, j in Q:
    E[i, j] = 255
    visited[i, j] = 1

nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
while Q:
    i, j = Q.popleft()
    for di, dj in nbrs:
        ni, nj = i+di, j+dj
        if 0 <= ni < E.shape[0] and 0 <= nj < E.shape[1]:
            if weak[ni, nj] and not visited[ni, nj]:
                E[ni, nj] = 255
                visited[ni, nj] = 1
                Q.append((ni, nj))

print("Edges (manual hysteresis) pixels:", int(E.sum()/255))

# --- Optional: OpenCV Canny for comparison ---
# Map our fractions to 8-bit thresholds by normalizing magnitude
mag8 = np.uint8(np.clip(255.0*mag/mag.max(), 0, 255))
low_cv  = int(T_low_frac*255)
high_cv = int(T_high_frac*255)
E_cv = cv2.Canny(Is, low_cv, high_cv, L2gradient=True)
print("Edges (cv2.Canny) pixels:", int(E_cv.sum()/255))

# --- Build 2x3 grid for the lecture ---
def to3(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
I_blur = Is
mag_v  = np.uint8(255.0*mag/mag.max())
nms_v  = np.uint8(255.0*mag_nms/(mag_nms.max()+1e-12))

# Color-code thresholds: black=non-edge, gray=weak, white=strong
th_map = np.zeros_like(I, dtype=np.uint8)
th_map[weak] = 128
th_map[strong] = 255

row1 = np.hstack([to3(I), to3(I_blur), to3(mag_v)])
row2 = np.hstack([to3(nms_v), to3(th_map), to3(E)])
grid = np.vstack([row1, row2])

cv2.imwrite("figs/canny_grid.png", grid)

import cv2, numpy as np, os
os.makedirs("figs", exist_ok=True)

# --- Read grayscale ---
img_path = "figs/sobel_input.jpg"   # <-- replace this
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
assert gray is not None, "Could not read input image"

# --- Sobel kernels (3x3) ---
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
Gy = np.array([[-1,-2,-1],
               [ 0, 0, 0],
               [ 1, 2, 1]], dtype=np.float32)

# --- Convolution (32F to keep precision) ---
Sx = cv2.filter2D(gray, cv2.CV_32F, Gx, borderType=cv2.BORDER_DEFAULT)
Sy = cv2.filter2D(gray, cv2.CV_32F, Gy, borderType=cv2.BORDER_DEFAULT)

# --- Separable implementation (optional parity check) ---
h_smooth = np.array([1,2,1], dtype=np.float32)   # [1,2,1]
h_diff   = np.array([-1,0,1], dtype=np.float32)  # [-1,0,1]
# Sx ≈ smooth vertically then diff horizontally:
Sx_sep = cv2.sepFilter2D(gray, cv2.CV_32F, h_diff, h_smooth)
# Sy ≈ diff vertically then smooth horizontally:
Sy_sep = cv2.sepFilter2D(gray, cv2.CV_32F, h_smooth, h_diff)
print("max|Sx-Sx_sep|:", float(np.max(np.abs(Sx - Sx_sep))))
print("max|Sy-Sy_sep|:", float(np.max(np.abs(Sy - Sy_sep))))

# --- Magnitude and direction (unsigned orientation in [0,180)) ---
mag = np.hypot(Sx, Sy)  # sqrt(Sx^2 + Sy^2)
ang = (np.degrees(np.arctan2(Sy, Sx)) % 180.0)  # orientation modulo 180

# --- Normalize for display ---
def norm8(x):
    x = np.abs(x)
    x = x / (x.max() + 1e-12)
    return np.uint8(np.round(x * 255.0))

Sx_v = norm8(Sx)
Sy_v = norm8(Sy)
mag_v = norm8(mag)

# --- Direction visualization in HSV (H=angle, S=V=mag) ---
H = np.uint8(np.round(ang))                 # 0..179 (close enough for HSV H)
S = mag_v.copy()                            # saturation ~ strength
V = mag_v.copy()                            # value ~ strength
hsv = cv2.merge([H, S, V])                  # OpenCV HSV expects H:0..179
dir_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# --- Simple threshold on magnitude ---
T = 0.25 * mag.max()
edges = (mag >= T).astype(np.uint8) * 255

# --- Make a 2x3 grid for lecture slide ---
to3 = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
row1 = np.hstack([to3(gray), to3(Sx_v), to3(Sy_v)])
row2 = np.hstack([to3(mag_v), dir_bgr, to3(edges)])
grid = np.vstack([row1, row2])

cv2.imwrite("figs/sobel_grid.png", grid)

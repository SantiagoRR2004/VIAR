import numpy as np, cv2, os

currentDirectory = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(currentDirectory, "imgs"), exist_ok=True)

img_path = os.path.join(currentDirectory, "input.jpg")
bgr = cv2.imread(img_path)
assert bgr is not None, "Could not read input image"

# (1) OpenCV conversion
hsv_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # H:[0,179], S,V:[0,255]

# (2) Manual RGB->HSV (match OpenCV scaling)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
maxc = np.max(rgb, axis=-1)
minc = np.min(rgb, axis=-1)
delta = maxc - minc

V = maxc
S = np.zeros_like(V)
nonzero_v = V > 1e-12
S[nonzero_v] = delta[nonzero_v] / V[nonzero_v]

H = np.zeros_like(V)

# Avoid division by zero
nz = delta > 1e-12
r_eq = (maxc == R) & nz
g_eq = (maxc == G) & nz
b_eq = (maxc == B) & nz

H[r_eq] = ((G[r_eq] - B[r_eq]) / delta[r_eq]) % 6.0
H[g_eq] = ((B[g_eq] - R[g_eq]) / delta[g_eq]) + 2.0
H[b_eq] = ((R[b_eq] - G[b_eq]) / delta[b_eq]) + 4.0
H_deg = (H * 60.0) % 360.0

# Match OpenCV ranges
H_u8 = np.round(H_deg / 2.0).astype(np.uint8)  # 0..179
S_u8 = np.round(S * 255.0).astype(np.uint8)  # 0..255
V_u8 = np.round(V * 255.0).astype(np.uint8)
hsv_manual = cv2.merge([H_u8, S_u8, V_u8])


# (3) Visualize H/S/V as grayscale and grid with original
def gray_to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# nice scaling for H visualization to 0..255
H_vis = np.uint8((H_u8.astype(np.float32) * (255.0 / 179.0)))
S_vis, V_vis = S_u8.copy(), V_u8.copy()

rgb_vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # for nicer color in docs
grid_top = np.hstack([rgb_vis, gray_to_bgr(H_vis)])
grid_bot = np.hstack([gray_to_bgr(S_vis), gray_to_bgr(V_vis)])
grid = np.vstack([grid_top, grid_bot])

cv2.imwrite(
    os.path.join(currentDirectory, "imgs", "rgb_hsv_channels.png"),
    cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
)

# (4) Algorithm parity check (abs diff per channel, amplified for visibility)
diff = cv2.absdiff(hsv_cv, hsv_manual)
# amplify to make tiny diffs visible, then clip
diff_amp = np.clip(diff.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
cv2.imwrite(
    os.path.join(currentDirectory, "imgs", "hsv_cv_vs_manual_diff.png"), diff_amp
)

# (5) Hue-driven recoloring (set S=255, V=255)
h_only = hsv_cv.copy()
h_only[..., 1:] = 255
hue_recolor = cv2.cvtColor(h_only, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join(currentDirectory, "imgs", "hue_recolor.png"), hue_recolor)

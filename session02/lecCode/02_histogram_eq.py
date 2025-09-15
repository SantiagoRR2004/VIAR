import cv2, numpy as np, os
#os.makedirs("imgs", exist_ok=True)

# --- Read & prep ---
img_path = "imgs/input_low_contrast.jpg"    
bgr = cv2.imread(img_path)
assert bgr is not None, "Could not read input image"
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# --- Global histogram equalization (grayscale) ---
he = cv2.equalizeHist(gray)

# --- CLAHE (adaptive, grayscale) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_gray = clahe.apply(gray)

# --- Grid: original vs HE vs CLAHE (grayscale) ---
to3 = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
grid_gray = np.hstack([to3(gray), to3(he), to3(clahe_gray)])
cv2.imwrite("imgs/histeq_grid.png", grid_gray)

# --- CLAHE on color (apply to L channel in LAB) ---
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
L2 = clahe.apply(L)
lab2 = cv2.merge([L2, A, B])
bgr_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

color_side_by_side = np.hstack([bgr, bgr_clahe])
cv2.imwrite("imgs/clahe_color_comparison.png", color_side_by_side)
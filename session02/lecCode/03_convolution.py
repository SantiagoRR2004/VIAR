import numpy as np
import cv2

# Example image (4x4)
I = np.array([
    [1, 2, 3, 4],
    [0, 1, 0, 1],
    [2, 3, 2, 3],
    [1, 0, 1, 0]
], dtype=np.float32)

# Example kernel (3x3 vertical edge detector)
K = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# --- Manual convolution (valid region only) ---
H, W = I.shape
Kh, Kw = K.shape
H_out, W_out = H - Kh + 1, W - Kw + 1

O_manual = np.zeros((H_out, W_out), dtype=np.float32)
for i in range(H_out):
    for j in range(W_out):
        patch = I[i:i+Kh, j:j+Kw]
        O_manual[i, j] = np.sum(patch * K)

print("Manual convolution:\n", O_manual)

# --- Using OpenCV ---
O_cv = cv2.filter2D(I, -1, K, borderType=cv2.BORDER_CONSTANT)
print("OpenCV filter2D:\n", O_cv)


# ---------_Benchmark


import numpy as np
import cv2
import time

def benchmark_conv(image_size, kernel_size):
    I = np.random.rand(image_size, image_size).astype(np.float32)
    K = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

    # OpenCV filter2D (optimized direct method)
    start = time.time()
    cv2.filter2D(I, -1, K)
    t = time.time() - start
    return t

sizes = [64, 128, 256, 512, 1024]
kernel = 15

for s in sizes:
    t = benchmark_conv(s, kernel)
    print(f"Image {s}x{s}, kernel {kernel}x{kernel}: {t:.4f} sec")

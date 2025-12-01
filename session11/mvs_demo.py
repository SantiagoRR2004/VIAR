"""
Demo: Classical Multi-View Stereo (MVS)
========================================
This demo illustrates the core MVS pipeline:
1. Start from sparse SfM result (cameras + sparse points)
2. Build cost volume using photometric consistency (NCC)
3. Compute per-view depth maps
4. Fuse depth maps into dense 3D point cloud
5. Compare sparse vs dense reconstruction

Author: Demo for VIAR Course
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage import gaussian_filter

np.random.seed(42)


# ============================================================================
# 1. SYNTHETIC SCENE WITH TEXTURE
# ============================================================================


def generate_textured_scene(img_width=320, img_height=240):
    """
    Generate a synthetic textured 3D scene.
    Creates a simple textured plane with depth variation.
    """
    # Create a textured surface (z varies as a function of x, y)
    x = np.linspace(-2, 2, img_width // 4)
    y = np.linspace(-1.5, 1.5, img_height // 4)
    X, Y = np.meshgrid(x, y)

    # Depth variation: bumpy surface
    Z = 5.0 + 0.5 * np.sin(2 * X) * np.cos(2 * Y)

    # Create 3D points
    points_3d = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    return points_3d, X, Y, Z


def create_synthetic_texture(width=320, height=240):
    """
    Create a synthetic texture image (Perlin-like noise).
    """
    # Generate random noise
    noise = np.random.rand(height // 8, width // 8)

    # Upsample with smoothing
    texture = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
    texture = gaussian_filter(texture, sigma=2.0)

    # Normalize to [0, 255]
    texture = (
        (texture - texture.min()) / (texture.max() - texture.min()) * 255
    ).astype(np.uint8)

    # Add some patterns
    for i in range(5):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        radius = np.random.randint(10, 30)
        cv2.circle(texture, (x, y), radius, 200, -1)

    return texture


def render_textured_image(points_3d, texture, R, t, K, img_width=320, img_height=240):
    """
    Render a textured image from given camera pose.
    """
    # Project 3D points to 2D
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_2d_hom = (P @ points_3d_hom.T).T

    # Perspective division
    depths = points_2d_hom[:, 2]
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

    # Create rendered image
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    depth_map = np.full((img_height, img_width), np.inf)

    # Simple Z-buffer rendering
    for i, (x, y, d) in enumerate(zip(points_2d[:, 0], points_2d[:, 1], depths)):
        xi, yi = int(round(x)), int(round(y))

        if 0 <= xi < img_width and 0 <= yi < img_height and d > 0:
            if d < depth_map[yi, xi]:
                depth_map[yi, xi] = d
                # Sample texture
                texture_x = int((i % texture.shape[1]))
                texture_y = int((i // texture.shape[1]) % texture.shape[0])
                image[yi, xi] = texture[texture_y, texture_x]

    # Fill holes with interpolation
    mask = depth_map != np.inf
    if np.sum(mask) > 0:
        image = cv2.inpaint(image, (~mask).astype(np.uint8), 3, cv2.INPAINT_NS)

    return image, depth_map


def create_camera_matrix(focal_length=400, img_width=320, img_height=240):
    """Create intrinsic camera matrix K"""
    cx, cy = img_width / 2, img_height / 2
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    return K


# ============================================================================
# 2. NORMALIZED CROSS-CORRELATION (NCC)
# ============================================================================


def compute_ncc(patch1, patch2):
    """
    Compute Normalized Cross-Correlation between two patches.

    NCC(W1, W2) = Σ(W1 - mean(W1))(W2 - mean(W2)) / (||W1 - mean(W1)|| ||W2 - mean(W2)||)

    Returns value in [-1, 1], where 1 is perfect match.
    """
    if patch1.size == 0 or patch2.size == 0:
        return -1.0

    patch1_flat = patch1.flatten().astype(np.float32)
    patch2_flat = patch2.flatten().astype(np.float32)

    # Center patches
    patch1_centered = patch1_flat - np.mean(patch1_flat)
    patch2_centered = patch2_flat - np.mean(patch2_flat)

    # Compute NCC
    norm1 = np.linalg.norm(patch1_centered)
    norm2 = np.linalg.norm(patch2_centered)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return -1.0

    ncc = np.dot(patch1_centered, patch2_centered) / (norm1 * norm2)

    return ncc


def extract_patch(image, x, y, patch_size=7):
    """
    Extract a patch centered at (x, y) from image.
    """
    h, w = image.shape
    half_size = patch_size // 2

    x1 = max(0, x - half_size)
    x2 = min(w, x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(h, y + half_size + 1)

    return image[y1:y2, x1:x2]


# ============================================================================
# 3. DEPTH MAP COMPUTATION
# ============================================================================


def compute_depth_map_mvs(
    ref_image,
    support_images,
    R_ref,
    t_ref,
    support_cameras,
    K,
    depth_min=3.0,
    depth_max=7.0,
    depth_samples=64,
    patch_size=7,
):
    """
    Compute depth map for reference view using MVS.

    For each pixel in reference view:
      1. Sample depth hypotheses
      2. For each depth, project to support views
      3. Compute NCC cost
      4. Select depth with best (highest) NCC score
    """
    h, w = ref_image.shape
    depth_map = np.zeros((h, w))
    confidence_map = np.zeros((h, w))

    # Depth hypotheses
    depths = np.linspace(depth_min, depth_max, depth_samples)

    print(f"  Computing depth map ({h} x {w}) with {depth_samples} depth samples...")

    # Process in blocks for efficiency
    step = 4  # Process every 4th pixel for speed

    for y in range(patch_size, h - patch_size, step):
        if y % 20 == 0:
            print(f"    Row {y}/{h}...")

        for x in range(patch_size, w - patch_size, step):
            # Get reference patch
            ref_patch = extract_patch(ref_image, x, y, patch_size)

            # Build cost volume for this pixel
            costs = np.zeros(depth_samples)

            for d_idx, depth in enumerate(depths):
                # 3D point in reference camera coordinates
                # x_cam = K^{-1} * [x, y, 1]^T * depth
                x_norm = (x - K[0, 2]) / K[0, 0]
                y_norm = (y - K[1, 2]) / K[1, 1]
                X_ref = np.array([x_norm * depth, y_norm * depth, depth])

                # Transform to world coordinates
                X_world = R_ref.T @ (X_ref - t_ref)

                # Accumulate NCC scores from support views
                ncc_scores = []

                for (R_sup, t_sup), sup_image in zip(support_cameras, support_images):
                    # Transform to support camera
                    X_sup = R_sup @ X_world + t_sup

                    # Project to support image
                    x_sup = K @ X_sup
                    if x_sup[2] <= 0:
                        continue

                    x_sup = x_sup[:2] / x_sup[2]
                    x_sup_i, y_sup_i = int(round(x_sup[0])), int(round(x_sup[1]))

                    # Check bounds
                    if (
                        patch_size <= x_sup_i < w - patch_size
                        and patch_size <= y_sup_i < h - patch_size
                    ):

                        # Extract support patch
                        sup_patch = extract_patch(
                            sup_image, x_sup_i, y_sup_i, patch_size
                        )

                        # Compute NCC
                        ncc = compute_ncc(ref_patch, sup_patch)
                        ncc_scores.append(ncc)

                # Average NCC across support views (higher is better)
                if len(ncc_scores) > 0:
                    costs[d_idx] = np.mean(ncc_scores)
                else:
                    costs[d_idx] = -1.0

            # Select depth with highest NCC score
            best_idx = np.argmax(costs)
            best_depth = depths[best_idx]
            best_score = costs[best_idx]

            depth_map[y, x] = best_depth
            confidence_map[y, x] = best_score

    # Interpolate to fill gaps
    mask = depth_map > 0
    if np.sum(mask) > 100:
        # Simple nearest-neighbor interpolation
        from scipy.interpolate import griddata

        points = np.argwhere(mask)
        values = depth_map[mask]

        grid_y, grid_x = np.mgrid[0:h, 0:w]
        depth_map_filled = griddata(points, values, (grid_y, grid_x), method="nearest")

        # Only fill where we had no data
        depth_map = np.where(mask, depth_map, depth_map_filled)

    return depth_map, confidence_map


# ============================================================================
# 4. DEPTH MAP TO POINT CLOUD
# ============================================================================


def depth_map_to_pointcloud(
    depth_map, confidence_map, image, R, t, K, confidence_threshold=0.3
):
    """
    Convert depth map to 3D point cloud.
    """
    h, w = depth_map.shape
    points_3d = []
    colors = []

    for y in range(h):
        for x in range(w):
            depth = depth_map[y, x]
            confidence = confidence_map[y, x]

            if depth > 0 and confidence > confidence_threshold:
                # Back-project to 3D in camera coordinates
                x_norm = (x - K[0, 2]) / K[0, 0]
                y_norm = (y - K[1, 2]) / K[1, 1]
                X_cam = np.array([x_norm * depth, y_norm * depth, depth])

                # Transform to world coordinates
                X_world = R.T @ (X_cam - t)

                points_3d.append(X_world)
                colors.append(image[y, x])

    return np.array(points_3d), np.array(colors)


# ============================================================================
# 5. VISUALIZATION
# ============================================================================


def visualize_mvs_results(
    images,
    depth_maps,
    confidence_maps,
    sparse_points,
    dense_points,
    cameras,
    save_path="mvs_demo_results.png",
):
    """
    Visualize MVS results.
    """
    fig = plt.figure(figsize=(18, 12))

    n_views = len(images)

    # Row 1: Input images
    for i in range(n_views):
        ax = plt.subplot(4, n_views, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Camera {i+1} - Input")
        ax.axis("off")

    # Row 2: Depth maps
    for i in range(n_views):
        ax = plt.subplot(4, n_views, n_views + i + 1)
        im = ax.imshow(depth_maps[i], cmap="jet")
        ax.set_title(f"Camera {i+1} - Depth Map")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Confidence maps
    for i in range(n_views):
        ax = plt.subplot(4, n_views, 2 * n_views + i + 1)
        im = ax.imshow(confidence_maps[i], cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"Camera {i+1} - Confidence (NCC)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 4: 3D reconstructions

    # Sparse reconstruction (simulated SfM output)
    ax1 = plt.subplot(4, n_views, 3 * n_views + 1, projection="3d")
    if len(sparse_points) > 0:
        ax1.scatter(
            sparse_points[:, 0],
            sparse_points[:, 1],
            sparse_points[:, 2],
            c="blue",
            s=5,
            alpha=0.5,
            label="Sparse SfM",
        )

    # Plot cameras
    for i, (R, t) in enumerate(cameras):
        C = -R.T @ t
        ax1.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Sparse SfM Output")
    ax1.legend()

    # Dense reconstruction (MVS output)
    ax2 = plt.subplot(4, n_views, 3 * n_views + 2, projection="3d")
    if len(dense_points) > 0:
        # Subsample for visualization
        step = max(1, len(dense_points) // 5000)
        sample = dense_points[::step]

        ax2.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            c="green",
            s=1,
            alpha=0.3,
            label="Dense MVS",
        )

    # Plot cameras
    for i, (R, t) in enumerate(cameras):
        C = -R.T @ t
        ax2.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Dense MVS Output")
    ax2.legend()

    # Side-by-side comparison
    ax3 = plt.subplot(4, n_views, 3 * n_views + 3, projection="3d")
    if len(sparse_points) > 0:
        ax3.scatter(
            sparse_points[:, 0],
            sparse_points[:, 1],
            sparse_points[:, 2],
            c="blue",
            s=10,
            alpha=0.6,
            label=f"Sparse ({len(sparse_points)} pts)",
        )

    if len(dense_points) > 0:
        step = max(1, len(dense_points) // 5000)
        sample = dense_points[::step]
        ax3.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            c="green",
            s=1,
            alpha=0.2,
            label=f"Dense ({len(dense_points)} pts)",
        )

    for i, (R, t) in enumerate(cameras):
        C = -R.T @ t
        ax3.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Sparse vs Dense Comparison")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Results saved to {save_path}")
    plt.show()


# ============================================================================
# MAIN DEMO
# ============================================================================


def main():
    """
    Main MVS demo pipeline.
    """
    print("=" * 70)
    print("CLASSICAL MULTI-VIEW STEREO (MVS) DEMO")
    print("=" * 70)

    # Parameters
    img_width, img_height = 320, 240

    # Step 1: Generate synthetic textured scene
    print("\n[Step 1] Generating synthetic textured scene...")
    points_3d, X, Y, Z = generate_textured_scene(img_width, img_height)
    texture = create_synthetic_texture(img_width, img_height)
    print(f"  - Created scene with {len(points_3d)} 3D points")

    # Step 2: Setup cameras
    print("\n[Step 2] Setting up camera array...")
    K = create_camera_matrix(
        focal_length=400, img_width=img_width, img_height=img_height
    )

    # Create 3 cameras at different positions
    cameras = []

    # Camera 1: frontal view
    R1 = np.eye(3)
    t1 = np.zeros(3)
    cameras.append((R1, t1))

    # Camera 2: slightly to the right
    angle = np.deg2rad(10)
    R2 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t2 = np.array([0.8, 0, 0])
    cameras.append((R2, t2))

    # Camera 3: slightly to the left
    angle = np.deg2rad(-10)
    R3 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t3 = np.array([-0.8, 0, 0])
    cameras.append((R3, t3))

    print(f"  - Created {len(cameras)} cameras")

    # Step 3: Render images
    print("\n[Step 3] Rendering images from each camera...")
    images = []
    ground_truth_depths = []

    for i, (R, t) in enumerate(cameras):
        image, depth = render_textured_image(
            points_3d, texture, R, t, K, img_width, img_height
        )
        images.append(image)
        ground_truth_depths.append(depth)
        print(f"  - Camera {i+1}: rendered {img_width}x{img_height} image")

    # Step 4: Simulate sparse SfM output
    print("\n[Step 4] Simulating sparse SfM output...")
    # Sample a sparse subset of points (e.g., 5% of total points)
    sparse_indices = np.random.choice(
        len(points_3d), size=len(points_3d) // 20, replace=False
    )
    sparse_points_3d = points_3d[sparse_indices]
    print(f"  - Sparse SfM: {len(sparse_points_3d)} points")

    # Step 5: Compute depth maps using MVS
    print("\n[Step 5] Computing depth maps using MVS...")
    depth_maps = []
    confidence_maps = []

    for i, (R_ref, t_ref) in enumerate(cameras):
        print(f"  Camera {i+1} as reference...")

        # Get support views (all other cameras)
        support_cameras = [cam for j, cam in enumerate(cameras) if j != i]
        support_images = [img for j, img in enumerate(images) if j != i]

        # Compute depth map
        depth_map, confidence_map = compute_depth_map_mvs(
            images[i],
            support_images,
            R_ref,
            t_ref,
            support_cameras,
            K,
            depth_min=3.0,
            depth_max=7.0,
            depth_samples=32,
            patch_size=7,
        )

        depth_maps.append(depth_map)
        confidence_maps.append(confidence_map)

        # Compute error vs ground truth
        gt_depth = ground_truth_depths[i]
        valid = (depth_map > 0) & (gt_depth != np.inf)
        if np.sum(valid) > 0:
            error = np.abs(depth_map[valid] - gt_depth[valid])
            print(f"    Mean depth error: {np.mean(error):.4f}")
            print(f"    Median depth error: {np.median(error):.4f}")

    # Step 6: Fuse depth maps into dense point cloud
    print("\n[Step 6] Fusing depth maps into dense point cloud...")
    all_dense_points = []
    all_colors = []

    for i, ((R, t), depth_map, conf_map, image) in enumerate(
        zip(cameras, depth_maps, confidence_maps, images)
    ):
        points, colors = depth_map_to_pointcloud(
            depth_map, conf_map, image, R, t, K, confidence_threshold=0.3
        )
        all_dense_points.append(points)
        all_colors.append(colors)
        print(f"  - Camera {i+1}: {len(points)} points")

    # Concatenate all points
    dense_points_3d = (
        np.vstack(all_dense_points) if len(all_dense_points) > 0 else np.array([])
    )
    print(f"  - Total dense points: {len(dense_points_3d)}")

    # Step 7: Visualize results
    print("\n[Step 7] Visualizing results...")
    visualize_mvs_results(
        images, depth_maps, confidence_maps, sparse_points_3d, dense_points_3d, cameras
    )

    # Statistics
    print("\n" + "=" * 70)
    print("MVS DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nReconstruction Statistics:")
    print(f"  - Sparse SfM points: {len(sparse_points_3d)}")
    print(f"  - Dense MVS points: {len(dense_points_3d)}")
    print(
        f"  - Densification ratio: {len(dense_points_3d) / len(sparse_points_3d):.1f}x"
    )
    print(f"  - Number of views: {len(cameras)}")

    print("\nKey takeaways:")
    print("  1. MVS starts from sparse SfM (cameras + sparse points)")
    print("  2. Photometric consistency (NCC) measures appearance similarity")
    print("  3. Cost volume aggregates matching scores across depth hypotheses")
    print("  4. Per-view depth maps capture dense surface geometry")
    print("  5. Depth fusion merges multiple views into consistent 3D model")
    print("  6. MVS produces 10-100x more points than sparse SfM")


if __name__ == "__main__":
    main()

"""
Demo: Classical Structure from Motion (SfM)
============================================
This demo illustrates the core SfM pipeline:
1. Feature detection and matching
2. Fundamental matrix estimation with RANSAC
3. Camera pose recovery (Essential matrix)
4. Triangulation to get 3D points
5. Bundle adjustment for refinement

Author: Demo for VIAR Course
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.optimize import least_squares
from matplotlib.patches import ConnectionPatch

np.random.seed(42)


# ============================================================================
# 1. SYNTHETIC SCENE GENERATION
# ============================================================================


def generate_synthetic_scene(n_points=50):
    """
    Generate a synthetic 3D scene with known ground truth.
    Points arranged in a rough cube to simulate a 3D structure.
    """
    # Create 3D points in a cube [-1, 1]^3 centered at (0, 0, 5)
    points_3d = np.random.uniform(-1, 1, (n_points, 3))
    points_3d[:, 2] += 5  # Move points away from cameras

    # Add some structure: corners of a cube
    corners = np.array(
        [
            [-1, -1, 4],
            [1, -1, 4],
            [1, 1, 4],
            [-1, 1, 4],
            [-1, -1, 6],
            [1, -1, 6],
            [1, 1, 6],
            [-1, 1, 6],
        ]
    )
    points_3d = np.vstack([points_3d, corners])

    return points_3d


def create_camera_matrix(focal_length=800, img_width=640, img_height=480):
    """Create intrinsic camera matrix K"""
    cx, cy = img_width / 2, img_height / 2
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    return K


def project_points(points_3d, R, t, K):
    """
    Project 3D points to 2D image plane.
    P = K[R|t]X
    """
    # Create projection matrix [R|t]
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    # Convert to homogeneous coordinates
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    # Project
    points_2d_hom = (P @ points_3d_hom.T).T

    # Convert to inhomogeneous
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

    return points_2d


def add_noise_to_points(points_2d, noise_std=1.0):
    """Add Gaussian noise to 2D points to simulate detection errors"""
    noise = np.random.normal(0, noise_std, points_2d.shape)
    return points_2d + noise


# ============================================================================
# 2. FUNDAMENTAL MATRIX ESTIMATION
# ============================================================================


def estimate_fundamental_matrix_ransac(pts1, pts2, threshold=3.0, n_iterations=1000):
    """
    Estimate fundamental matrix using RANSAC.

    The fundamental matrix F satisfies: x2^T F x1 = 0
    """
    n_points = len(pts1)
    best_F = None
    best_inliers = []
    best_score = 0

    for _ in range(n_iterations):
        # Sample 8 points
        indices = np.random.choice(n_points, 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Estimate F using 8-point algorithm
        F = estimate_fundamental_8point(sample_pts1, sample_pts2)

        if F is None:
            continue

        # Compute inliers
        inliers = compute_epipolar_inliers(pts1, pts2, F, threshold)
        score = np.sum(inliers)

        if score > best_score:
            best_score = score
            best_F = F
            best_inliers = inliers

    # Refine with all inliers
    if best_F is not None and np.sum(best_inliers) >= 8:
        inlier_pts1 = pts1[best_inliers]
        inlier_pts2 = pts2[best_inliers]
        best_F = estimate_fundamental_8point(inlier_pts1, inlier_pts2)

    return best_F, best_inliers


def estimate_fundamental_8point(pts1, pts2):
    """
    8-point algorithm for fundamental matrix estimation.
    Implements normalized 8-point algorithm.
    """
    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build constraint matrix A
    n = len(pts1_norm)
    A = np.zeros((n, 9))

    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0  # Set smallest singular value to zero
    F_norm = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F_norm @ T1

    return F / F[2, 2]  # Normalize


def normalize_points(pts):
    """Normalize points for numerical stability"""
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts_centered, axis=1))

    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm_hom = (T @ pts_hom.T).T
    pts_norm = pts_norm_hom[:, :2] / pts_norm_hom[:, 2:3]

    return pts_norm, T


def compute_epipolar_inliers(pts1, pts2, F, threshold):
    """
    Compute inliers based on epipolar constraint.
    """
    pts1_hom = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_hom = np.hstack([pts2, np.ones((len(pts2), 1))])

    # Compute epipolar error: |x2^T F x1|
    errors = np.abs(np.sum(pts2_hom * (F @ pts1_hom.T).T, axis=1))

    inliers = errors < threshold
    return inliers


# ============================================================================
# 3. CAMERA POSE RECOVERY
# ============================================================================


def essential_from_fundamental(F, K1, K2):
    """
    Compute essential matrix from fundamental matrix.
    E = K2^T F K1
    """
    E = K2.T @ F @ K1

    # Enforce essential matrix constraints (two equal singular values)
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1, 1, 0])  # Ideal essential matrix
    E = U @ np.diag(S) @ Vt

    return E


def recover_pose_from_essential(E, pts1, pts2, K):
    """
    Recover camera pose (R, t) from essential matrix.
    Tests 4 possible solutions and picks the one with points in front of cameras.
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Four possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    # Test each solution
    best_solution = None
    max_positive = 0

    for R, t in solutions:
        # Triangulate a few points and count positive depths
        pts_3d = triangulate_points_simple(
            pts1[:10], pts2[:10], np.eye(3), np.zeros(3), R, t, K
        )

        # Check depths in both cameras
        depths1 = pts_3d[:, 2]
        depths2 = (R @ pts_3d.T + t.reshape(3, 1))[2, :]

        n_positive = np.sum((depths1 > 0) & (depths2 > 0))

        if n_positive > max_positive:
            max_positive = n_positive
            best_solution = (R, t)

    return best_solution


# ============================================================================
# 4. TRIANGULATION
# ============================================================================


def triangulate_points_simple(pts1, pts2, R1, t1, R2, t2, K):
    """
    Triangulate 3D points from two views using DLT (Direct Linear Transform).
    """
    # Projection matrices
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    n_points = len(pts1)
    points_3d = np.zeros((n_points, 3))

    for i in range(n_points):
        # Build matrix A for DLT
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array(
            [
                x1 * P1[2, :] - P1[0, :],
                y1 * P1[2, :] - P1[1, :],
                x2 * P2[2, :] - P2[0, :],
                y2 * P2[2, :] - P2[1, :],
            ]
        )

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        X = X_hom[:3] / X_hom[3]

        points_3d[i] = X

    return points_3d


# ============================================================================
# 5. BUNDLE ADJUSTMENT
# ============================================================================


def bundle_adjustment(points_3d, points_2d_views, cameras, K):
    """
    Bundle adjustment: jointly optimize 3D points and camera poses.

    Minimizes reprojection error:
    min sum_{i,j} ||x_ij - π(R_j, t_j, X_i)||^2
    """
    n_cameras = len(cameras)
    n_points = len(points_3d)

    # Initial parameters
    # Format: [cam1_params, cam2_params, ..., point1, point2, ...]
    # Camera params: [rodrigues_rotation(3), translation(3)]
    params = []

    for R, t in cameras:
        rodrigues, _ = cv2.Rodrigues(R)
        params.extend(rodrigues.flatten())
        params.extend(t.flatten())

    for pt in points_3d:
        params.extend(pt)

    params = np.array(params)

    # Camera indices for each observation
    camera_indices = []
    point_indices = []
    points_2d = []

    for cam_idx, pts_2d in enumerate(points_2d_views):
        for pt_idx in range(len(pts_2d)):
            if pt_idx < n_points:  # Valid point
                camera_indices.append(cam_idx)
                point_indices.append(pt_idx)
                points_2d.append(pts_2d[pt_idx])

    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)

    # Optimize
    result = least_squares(
        bundle_adjustment_residuals,
        params,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
        verbose=0,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
    )

    # Extract optimized parameters
    optimized_params = result.x
    optimized_cameras = []

    for i in range(n_cameras):
        idx = i * 6
        rodrigues = optimized_params[idx : idx + 3]
        t = optimized_params[idx + 3 : idx + 6]
        R, _ = cv2.Rodrigues(rodrigues)
        optimized_cameras.append((R, t))

    optimized_points = optimized_params[n_cameras * 6 :].reshape(-1, 3)

    return optimized_cameras, optimized_points


def bundle_adjustment_residuals(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, K
):
    """
    Compute residuals for bundle adjustment.
    """
    # Extract camera parameters
    cameras = []
    for i in range(n_cameras):
        idx = i * 6
        rodrigues = params[idx : idx + 3]
        t = params[idx + 3 : idx + 6]
        R, _ = cv2.Rodrigues(rodrigues)
        cameras.append((R, t))

    # Extract 3D points
    points_3d = params[n_cameras * 6 :].reshape(-1, 3)

    # Compute reprojection errors
    residuals = []

    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        R, t = cameras[cam_idx]
        X = points_3d[pt_idx]

        # Project point
        X_cam = R @ X + t
        x_proj = K @ X_cam
        x_proj = x_proj[:2] / x_proj[2]

        # Compute error
        error = points_2d[i] - x_proj
        residuals.extend(error)

    return np.array(residuals)


# ============================================================================
# 6. VISUALIZATION
# ============================================================================


def visualize_sfm_results(
    points_3d_gt,
    points_3d_sfm,
    cameras,
    points_2d_views,
    inliers_list,
    save_path="sfm_demo_results.png",
):
    """
    Visualize SfM results: 2D correspondences and 3D reconstruction.
    """
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 2D correspondences (camera 1 and 2)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(
        points_2d_views[0][:, 0],
        points_2d_views[0][:, 1],
        c="blue",
        s=30,
        alpha=0.6,
        label="Camera 1",
    )
    ax1.set_title("Camera 1 - Feature Points")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(
        points_2d_views[1][:, 0],
        points_2d_views[1][:, 1],
        c="red",
        s=30,
        alpha=0.6,
        label="Camera 2",
    )
    ax2.set_title("Camera 2 - Feature Points")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 2: Matches
    ax3 = plt.subplot(2, 3, 3)
    # Show inliers
    inliers = (
        inliers_list[0]
        if len(inliers_list) > 0
        else np.ones(len(points_2d_views[0]), dtype=bool)
    )
    colors = ["green" if inlier else "red" for inlier in inliers[:20]]

    for i in range(min(20, len(points_2d_views[0]))):
        ax3.plot([0, 1], [i, i], "o-", color=colors[i], alpha=0.5)

    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-1, 20)
    ax3.set_title(
        f"Matches (green=inlier, red=outlier)\n{np.sum(inliers)} / {len(inliers)} inliers"
    )
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Cam 1", "Cam 2"])
    ax3.invert_yaxis()

    # Plot 3: Ground truth 3D
    ax4 = plt.subplot(2, 3, 4, projection="3d")
    ax4.scatter(
        points_3d_gt[:, 0],
        points_3d_gt[:, 1],
        points_3d_gt[:, 2],
        c="blue",
        s=30,
        alpha=0.6,
        label="Ground Truth",
    )

    # Plot cameras
    for i, (R, t) in enumerate(cameras):
        # Camera center: C = -R^T t
        C = -R.T @ t
        ax4.scatter(
            C[0],
            C[1],
            C[2],
            c="black",
            s=100,
            marker="^",
            label=f"Camera {i+1}" if i < 2 else "",
        )

        # Camera direction
        direction = R.T @ np.array([0, 0, 1])
        ax4.quiver(
            C[0],
            C[1],
            C[2],
            direction[0],
            direction[1],
            direction[2],
            length=0.5,
            color="gray",
            alpha=0.6,
        )

    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("Ground Truth 3D Scene")
    ax4.legend()

    # Plot 4: Reconstructed 3D (before BA)
    ax5 = plt.subplot(2, 3, 5, projection="3d")
    ax5.scatter(
        points_3d_sfm[:, 0],
        points_3d_sfm[:, 1],
        points_3d_sfm[:, 2],
        c="red",
        s=30,
        alpha=0.6,
        label="SfM (before BA)",
    )

    for i, (R, t) in enumerate(cameras[:2]):
        C = -R.T @ t
        ax5.scatter(C[0], C[1], C[2], c="black", s=100, marker="^")

    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")
    ax5.set_title("SfM Reconstruction (Initial)")
    ax5.legend()

    # Plot 5: Comparison overlay
    ax6 = plt.subplot(2, 3, 6, projection="3d")
    ax6.scatter(
        points_3d_gt[:, 0],
        points_3d_gt[:, 1],
        points_3d_gt[:, 2],
        c="blue",
        s=30,
        alpha=0.3,
        label="Ground Truth",
    )
    ax6.scatter(
        points_3d_sfm[:, 0],
        points_3d_sfm[:, 1],
        points_3d_sfm[:, 2],
        c="red",
        s=30,
        alpha=0.6,
        label="SfM",
    )

    for i, (R, t) in enumerate(cameras[:2]):
        C = -R.T @ t
        ax6.scatter(C[0], C[1], C[2], c="black", s=100, marker="^")

    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_zlabel("Z")
    ax6.set_title("Overlay: GT vs SfM")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Results saved to {save_path}")
    plt.show()


# ============================================================================
# MAIN DEMO
# ============================================================================


def main():
    """
    Main SfM demo pipeline.
    """
    print("=" * 70)
    print("CLASSICAL STRUCTURE FROM MOTION (SfM) DEMO")
    print("=" * 70)

    # Step 1: Generate synthetic scene
    print("\n[Step 1] Generating synthetic 3D scene...")
    points_3d_gt = generate_synthetic_scene(n_points=50)
    print(f"  - Created {len(points_3d_gt)} 3D points")

    # Step 2: Create camera setup
    print("\n[Step 2] Setting up cameras...")
    K = create_camera_matrix(focal_length=800, img_width=640, img_height=480)
    print(f"  - Camera intrinsics K:")
    print(f"    {K}")

    # Camera 1: identity (world frame)
    R1 = np.eye(3)
    t1 = np.zeros(3)

    # Camera 2: rotated and translated
    angle = np.deg2rad(15)
    R2 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t2 = np.array([1.5, 0, 0])

    cameras_gt = [(R1, t1), (R2, t2)]

    # Step 3: Project points to images
    print("\n[Step 3] Projecting 3D points to 2D images...")
    points_2d_cam1 = project_points(points_3d_gt, R1, t1, K)
    points_2d_cam2 = project_points(points_3d_gt, R2, t2, K)

    # Add noise
    points_2d_cam1_noisy = add_noise_to_points(points_2d_cam1, noise_std=0.5)
    points_2d_cam2_noisy = add_noise_to_points(points_2d_cam2, noise_std=0.5)

    print(f"  - Camera 1: {len(points_2d_cam1_noisy)} points")
    print(f"  - Camera 2: {len(points_2d_cam2_noisy)} points")

    # Step 4: Estimate fundamental matrix
    print("\n[Step 4] Estimating Fundamental Matrix with RANSAC...")
    F, inliers = estimate_fundamental_matrix_ransac(
        points_2d_cam1_noisy, points_2d_cam2_noisy, threshold=3.0, n_iterations=1000
    )
    print(f"  - Found {np.sum(inliers)} / {len(inliers)} inliers")
    print(f"  - Fundamental matrix F:")
    print(f"    {F}")

    # Step 5: Recover camera pose
    print("\n[Step 5] Recovering camera pose from Essential Matrix...")
    E = essential_from_fundamental(F, K, K)

    inlier_pts1 = points_2d_cam1_noisy[inliers]
    inlier_pts2 = points_2d_cam2_noisy[inliers]

    R_est, t_est = recover_pose_from_essential(E, inlier_pts1, inlier_pts2, K)
    print(f"  - Estimated rotation R:")
    print(f"    {R_est}")
    print(f"  - Estimated translation t: {t_est}")

    # Step 6: Triangulate 3D points
    print("\n[Step 6] Triangulating 3D points...")
    points_3d_sfm = triangulate_points_simple(
        inlier_pts1, inlier_pts2, R1, t1, R_est, t_est, K
    )
    print(f"  - Reconstructed {len(points_3d_sfm)} 3D points")

    # Compute reconstruction error
    gt_inliers = points_3d_gt[inliers]

    # Align reconstructed points to ground truth (account for scale ambiguity)
    scale = np.median(
        np.linalg.norm(gt_inliers, axis=1) / np.linalg.norm(points_3d_sfm, axis=1)
    )
    points_3d_sfm_scaled = points_3d_sfm * scale

    errors = np.linalg.norm(gt_inliers - points_3d_sfm_scaled, axis=1)
    print(f"  - Mean reconstruction error: {np.mean(errors):.4f}")
    print(f"  - Median reconstruction error: {np.median(errors):.4f}")

    # Step 7: Bundle Adjustment
    print("\n[Step 7] Running Bundle Adjustment...")
    cameras_init = [(R1, t1), (R_est, t_est)]
    points_2d_views = [inlier_pts1, inlier_pts2]

    cameras_opt, points_3d_opt = bundle_adjustment(
        points_3d_sfm, points_2d_views, cameras_init, K
    )

    # Scale optimized points
    scale_opt = np.median(
        np.linalg.norm(gt_inliers, axis=1) / np.linalg.norm(points_3d_opt, axis=1)
    )
    points_3d_opt_scaled = points_3d_opt * scale_opt

    errors_opt = np.linalg.norm(gt_inliers - points_3d_opt_scaled, axis=1)
    print(f"  - Mean error after BA: {np.mean(errors_opt):.4f}")
    print(f"  - Median error after BA: {np.median(errors_opt):.4f}")
    print(f"  - Improvement: {(np.mean(errors) - np.mean(errors_opt)):.4f}")

    # Step 8: Visualize
    print("\n[Step 8] Visualizing results...")
    visualize_sfm_results(
        gt_inliers, points_3d_sfm_scaled, cameras_gt, points_2d_views, [inliers]
    )

    print("\n" + "=" * 70)
    print("SfM DEMO COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Feature matching establishes 2D correspondences")
    print("  2. Fundamental matrix captures epipolar geometry")
    print("  3. Essential matrix enables camera pose recovery")
    print("  4. Triangulation reconstructs sparse 3D structure")
    print("  5. Bundle adjustment refines cameras and 3D points jointly")


if __name__ == "__main__":
    main()

"""
Complete Pipeline Demo: SfM → MVS
==================================
This demo shows the complete classical 3D reconstruction pipeline:
  Sparse SfM → Dense MVS

This illustrates how the two methods work together in practice.

Author: Demo for VIAR Course
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import functions from the individual demos
# (In practice, you would run this after ensuring both demo files are in the same directory)

np.random.seed(42)


def create_simple_scene():
    """Create a simple 3D scene for demonstration."""
    # Create a flat plane with some depth variation
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-1.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)

    # Add gentle surface undulation
    Z = 5.0 + 0.3 * np.sin(2 * X) * np.cos(2 * Y)

    points_3d = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points_3d


def create_camera_setup():
    """Create multiple camera poses."""
    cameras = []

    # Camera 1: frontal
    R1 = np.eye(3)
    t1 = np.zeros(3)
    cameras.append(("Camera 1 (frontal)", R1, t1))

    # Camera 2: right
    angle = np.deg2rad(15)
    R2 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t2 = np.array([1.2, 0, 0])
    cameras.append(("Camera 2 (right)", R2, t2))

    # Camera 3: left
    angle = np.deg2rad(-15)
    R3 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t3 = np.array([-1.2, 0, 0])
    cameras.append(("Camera 3 (left)", R3, t3))

    return cameras


def simulate_sfm_sparse_output(points_3d, sampling_rate=0.05):
    """
    Simulate sparse SfM output by sampling a subset of points.

    In real SfM, we only get points where there are strong features
    (corners, edges, textured regions).
    """
    n_sparse = int(len(points_3d) * sampling_rate)
    indices = np.random.choice(len(points_3d), n_sparse, replace=False)
    sparse_points = points_3d[indices]

    return sparse_points


def simulate_mvs_dense_output(points_3d, sparse_points, density_factor=0.6):
    """
    Simulate dense MVS output.

    MVS fills in the gaps between sparse SfM points using photometric
    consistency across multiple views.
    """
    # In this simulation, we'll use a larger subset of the original points
    # to represent the dense reconstruction
    n_dense = int(len(points_3d) * density_factor)

    # Sample more densely, but still not completely (some regions fail)
    indices = np.random.choice(len(points_3d), n_dense, replace=False)
    dense_points = points_3d[indices]

    # Add small noise to simulate reconstruction errors
    noise = np.random.normal(0, 0.02, dense_points.shape)
    dense_points_noisy = dense_points + noise

    return dense_points_noisy


def visualize_complete_pipeline(points_3d, sparse_points, dense_points, cameras):
    """
    Create a comprehensive visualization showing the complete pipeline.
    """
    fig = plt.figure(figsize=(20, 12))

    # Common view limits (for consistent comparison)
    xlim = [-3, 3]
    ylim = [-2.5, 2.5]
    zlim = [3, 7]

    def setup_3d_axis(ax, title):
        """Helper to setup 3D axis with consistent view."""
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.view_init(elev=20, azim=45)

    # ========================================
    # Row 1: Input - Multiple views concept
    # ========================================
    ax1 = plt.subplot(3, 4, 1, projection="3d")

    # Plot ground truth scene
    step = 10
    ax1.scatter(
        points_3d[::step, 0],
        points_3d[::step, 1],
        points_3d[::step, 2],
        c="lightgray",
        s=5,
        alpha=0.3,
        label="Scene",
    )

    # Plot cameras
    for name, R, t in cameras:
        C = -R.T @ t
        ax1.scatter(
            C[0],
            C[1],
            C[2],
            c="red",
            s=200,
            marker="^",
            edgecolors="black",
            linewidths=2,
        )

        # Camera viewing direction
        direction = R.T @ np.array([0, 0, 1.5])
        ax1.quiver(
            C[0],
            C[1],
            C[2],
            direction[0],
            direction[1],
            direction[2],
            color="red",
            alpha=0.6,
            arrow_length_ratio=0.3,
            linewidth=2,
        )

    setup_3d_axis(ax1, "Input: Multi-View Setup")
    ax1.legend(loc="upper right", fontsize=8)

    # ========================================
    # Row 1: Camera poses diagram
    # ========================================
    ax2 = plt.subplot(3, 4, 2)

    # Top-down view of camera arrangement
    for i, (name, R, t) in enumerate(cameras):
        C = -R.T @ t
        ax2.scatter(
            C[0], C[2], c="red", s=200, marker="^", edgecolors="black", linewidths=2
        )
        ax2.text(
            C[0], C[2] - 0.3, f"C{i+1}", ha="center", fontsize=10, fontweight="bold"
        )

        # Viewing direction
        direction = R.T @ np.array([0, 0, 1.0])
        ax2.arrow(
            C[0],
            C[2],
            direction[0],
            direction[2],
            head_width=0.15,
            head_length=0.2,
            fc="red",
            ec="red",
            alpha=0.6,
        )

    # Show scene center
    ax2.scatter(0, 5, c="blue", s=300, marker="o", alpha=0.3, label="Scene")

    ax2.set_xlabel("X (meters)", fontsize=10)
    ax2.set_ylabel("Z (depth, meters)", fontsize=10)
    ax2.set_title("Top-Down View: Camera Array", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axis("equal")

    # ========================================
    # Row 1: Pipeline diagram
    # ========================================
    ax3 = plt.subplot(3, 4, 3)
    ax3.axis("off")

    # Pipeline flowchart
    pipeline_text = """
    CLASSICAL 3D RECONSTRUCTION PIPELINE
    ═══════════════════════════════════════
    
    1. FEATURE EXTRACTION
       ↓ SIFT / SURF features
    
    2. FEATURE MATCHING
       ↓ Correspondences
    
    3. FUNDAMENTAL MATRIX
       ↓ RANSAC (epipolar geometry)
    
    4. CAMERA POSE RECOVERY
       ↓ Essential matrix → R, t
    
    5. TRIANGULATION
       ↓ Sparse 3D points
    
    6. BUNDLE ADJUSTMENT
       ↓ Optimize cameras + points
    
    → SPARSE RECONSTRUCTION
    
    7. COST VOLUME (NCC)
       ↓ Photometric consistency
    
    8. DEPTH ESTIMATION
       ↓ Per-pixel depths
    
    9. DEPTH FUSION
       ↓ Merge views
    
    → DENSE RECONSTRUCTION
    """

    ax3.text(
        0.1,
        0.5,
        pipeline_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax3.set_title("Processing Pipeline", fontsize=12, fontweight="bold")

    # ========================================
    # Row 1: Statistics panel
    # ========================================
    ax4 = plt.subplot(3, 4, 4)
    ax4.axis("off")

    stats_text = f"""
    RECONSTRUCTION STATISTICS
    ═══════════════════════════
    
    Scene Information:
    • Total points: {len(points_3d):,}
    • Scene extent: {xlim[1] - xlim[0]:.1f}m × {ylim[1] - ylim[0]:.1f}m
    • Depth range: {zlim[0]:.1f}m - {zlim[1]:.1f}m
    
    Camera Setup:
    • Number of views: {len(cameras)}
    • Baseline: ~{2.4:.1f}m
    • Focal length: 400px
    
    SfM Output (Sparse):
    • Reconstructed: {len(sparse_points):,} points
    • Sampling rate: {100*len(sparse_points)/len(points_3d):.1f}%
    • Feature-based
    
    MVS Output (Dense):
    • Reconstructed: {len(dense_points):,} points
    • Sampling rate: {100*len(dense_points)/len(points_3d):.1f}%
    • Photometric-based
    
    Densification:
    • Factor: {len(dense_points)/len(sparse_points):.1f}×
    • Improvement: {len(dense_points)-len(sparse_points):,} points
    """

    ax4.text(
        0.1,
        0.5,
        stats_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )
    ax4.set_title("Statistics", fontsize=12, fontweight="bold")

    # ========================================
    # Row 2: Ground Truth
    # ========================================
    ax5 = plt.subplot(3, 4, 5, projection="3d")

    step = 5
    ax5.scatter(
        points_3d[::step, 0],
        points_3d[::step, 1],
        points_3d[::step, 2],
        c=points_3d[::step, 2],
        cmap="viridis",
        s=10,
        alpha=0.5,
    )

    # Add cameras
    for name, R, t in cameras:
        C = -R.T @ t
        ax5.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax5, "Ground Truth Scene")

    # ========================================
    # Row 2: Sparse SfM Output
    # ========================================
    ax6 = plt.subplot(3, 4, 6, projection="3d")

    ax6.scatter(
        sparse_points[:, 0],
        sparse_points[:, 1],
        sparse_points[:, 2],
        c="blue",
        s=30,
        alpha=0.7,
        label=f"Sparse ({len(sparse_points)} pts)",
    )

    # Add cameras
    for name, R, t in cameras:
        C = -R.T @ t
        ax6.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax6, "Stage 1: Sparse SfM Output")
    ax6.legend(loc="upper right", fontsize=8)

    # ========================================
    # Row 2: Dense MVS Output
    # ========================================
    ax7 = plt.subplot(3, 4, 7, projection="3d")

    step_dense = max(1, len(dense_points) // 3000)
    ax7.scatter(
        dense_points[::step_dense, 0],
        dense_points[::step_dense, 1],
        dense_points[::step_dense, 2],
        c="green",
        s=5,
        alpha=0.4,
        label=f"Dense ({len(dense_points)} pts)",
    )

    # Add cameras
    for name, R, t in cameras:
        C = -R.T @ t
        ax7.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax7, "Stage 2: Dense MVS Output")
    ax7.legend(loc="upper right", fontsize=8)

    # ========================================
    # Row 2: Overlay Comparison
    # ========================================
    ax8 = plt.subplot(3, 4, 8, projection="3d")

    # Sparse in blue
    ax8.scatter(
        sparse_points[:, 0],
        sparse_points[:, 1],
        sparse_points[:, 2],
        c="blue",
        s=40,
        alpha=0.8,
        label="Sparse",
        edgecolors="darkblue",
        linewidths=0.5,
    )

    # Dense in green (subsampled)
    step_dense = max(1, len(dense_points) // 2000)
    ax8.scatter(
        dense_points[::step_dense, 0],
        dense_points[::step_dense, 1],
        dense_points[::step_dense, 2],
        c="green",
        s=3,
        alpha=0.3,
        label="Dense",
    )

    # Add cameras
    for name, R, t in cameras:
        C = -R.T @ t
        ax8.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax8, "Overlay: Sparse vs Dense")
    ax8.legend(loc="upper right", fontsize=8)

    # ========================================
    # Row 3: Different viewing angles
    # ========================================

    # View 1: Side view
    ax9 = plt.subplot(3, 4, 9, projection="3d")
    step_dense = max(1, len(dense_points) // 3000)
    ax9.scatter(
        dense_points[::step_dense, 0],
        dense_points[::step_dense, 1],
        dense_points[::step_dense, 2],
        c=dense_points[::step_dense, 2],
        cmap="viridis",
        s=5,
        alpha=0.5,
    )

    for name, R, t in cameras:
        C = -R.T @ t
        ax9.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax9, "Dense MVS - Side View")
    ax9.view_init(elev=5, azim=0)

    # View 2: Top view
    ax10 = plt.subplot(3, 4, 10, projection="3d")
    ax10.scatter(
        dense_points[::step_dense, 0],
        dense_points[::step_dense, 1],
        dense_points[::step_dense, 2],
        c=dense_points[::step_dense, 2],
        cmap="viridis",
        s=5,
        alpha=0.5,
    )

    for name, R, t in cameras:
        C = -R.T @ t
        ax10.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax10, "Dense MVS - Top View")
    ax10.view_init(elev=90, azim=0)

    # View 3: Frontal view
    ax11 = plt.subplot(3, 4, 11, projection="3d")
    ax11.scatter(
        dense_points[::step_dense, 0],
        dense_points[::step_dense, 1],
        dense_points[::step_dense, 2],
        c=dense_points[::step_dense, 2],
        cmap="viridis",
        s=5,
        alpha=0.5,
    )

    for name, R, t in cameras:
        C = -R.T @ t
        ax11.scatter(C[0], C[1], C[2], c="red", s=100, marker="^")

    setup_3d_axis(ax11, "Dense MVS - Frontal View")
    ax11.view_init(elev=0, azim=90)

    # ========================================
    # Row 3: Point density comparison
    # ========================================
    ax12 = plt.subplot(3, 4, 12)

    # Create density histogram
    labels = ["Sparse SfM", "Dense MVS", "Ground Truth"]
    counts = [len(sparse_points), len(dense_points), len(points_3d)]
    colors = ["blue", "green", "gray"]

    bars = ax12.bar(
        labels, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax12.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax12.set_ylabel("Number of 3D Points", fontsize=11)
    ax12.set_title("Point Density Comparison", fontsize=12, fontweight="bold")
    ax12.grid(axis="y", alpha=0.3)

    # Add densification annotation
    ax12.annotate(
        "",
        xy=(1, len(dense_points)),
        xytext=(0, len(sparse_points)),
        arrowprops=dict(arrowstyle="->", lw=2, color="red"),
    )
    ax12.text(
        0.5,
        (len(sparse_points) + len(dense_points)) / 2,
        f"{len(dense_points)/len(sparse_points):.1f}× denser",
        ha="center",
        fontsize=10,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig("complete_pipeline_comparison.png", dpi=200, bbox_inches="tight")
    print("Complete pipeline visualization saved to 'complete_pipeline_comparison.png'")
    plt.show()


def main():
    """
    Main function demonstrating the complete SfM → MVS pipeline.
    """
    print("=" * 70)
    print("COMPLETE CLASSICAL 3D RECONSTRUCTION PIPELINE")
    print("Structure from Motion (SfM) → Multi-View Stereo (MVS)")
    print("=" * 70)

    print("\n[Step 1] Creating synthetic scene...")
    points_3d = create_simple_scene()
    print(f"  - Scene: {len(points_3d)} points")

    print("\n[Step 2] Setting up camera array...")
    cameras = create_camera_setup()
    print(f"  - Cameras: {len(cameras)} views")
    for name, R, t in cameras:
        print(f"    • {name}")

    print("\n[Step 3] Simulating sparse SfM reconstruction...")
    sparse_points = simulate_sfm_sparse_output(points_3d, sampling_rate=0.05)
    print(
        f"  - Sparse points: {len(sparse_points)} ({100*len(sparse_points)/len(points_3d):.1f}% of scene)"
    )
    print(f"  - These represent feature-rich regions (corners, edges)")

    print("\n[Step 4] Simulating dense MVS reconstruction...")
    dense_points = simulate_mvs_dense_output(
        points_3d, sparse_points, density_factor=0.6
    )
    print(
        f"  - Dense points: {len(dense_points)} ({100*len(dense_points)/len(points_3d):.1f}% of scene)"
    )
    print(f"  - Densification factor: {len(dense_points)/len(sparse_points):.1f}×")
    print(f"  - Additional points: {len(dense_points) - len(sparse_points)}")

    print("\n[Step 5] Visualizing complete pipeline...")
    visualize_complete_pipeline(points_3d, sparse_points, dense_points, cameras)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print("  1. SfM produces SPARSE reconstruction from feature correspondences")
    print("  2. MVS produces DENSE reconstruction using photometric consistency")
    print("  3. MVS fills in ~10-20× more points than SfM")
    print("  4. Both methods are complementary and often used together")
    print("  5. Classical methods still form the basis of modern 3D reconstruction")

    print("\nNext steps:")
    print("  - Run demo_sfm_classical.py for detailed SfM implementation")
    print("  - Run demo_mvs_classical.py for detailed MVS implementation")
    print("  - Explore how deep learning improves on these classical methods")


if __name__ == "__main__":
    main()

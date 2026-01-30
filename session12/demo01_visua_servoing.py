"""
Demo 01: Visual Servoing - Image-Based Visual Servoing (IBVS)
=============================================================

This demo illustrates the classical visual servoing approach where a camera
(or robot end-effector with camera) is controlled to reach a desired pose
by minimizing errors in image feature space.

Key Concepts:
- Image features (point coordinates in image plane)
- Image Jacobian (relates feature velocity to camera velocity)
- Feedback control law: v = -λ * J^+ * e

The demo simulates a camera viewing 4 corner points of a square target.
The goal is to move the camera from an initial pose to align with the
desired image of the target.

Historical Context:
- Visual servoing emerged in the 1990s as a way to close the control loop
  through visual feedback rather than relying on precise calibration.
- IBVS works directly in image space, avoiding 3D reconstruction errors.
- PBVS (Position-Based) reconstructs 3D pose first, then applies Cartesian control.

Reference: Chaumette & Hutchinson, "Visual Servo Control" (2006, 2007)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation

# ============================================================================
# Camera and Projection Model
# ============================================================================


class PinholeCamera:
    """Simple pinhole camera model for visual servoing simulation."""

    def __init__(self, fx=800, fy=800, cx=320, cy=240, img_width=640, img_height=480):
        """
        Initialize camera with intrinsic parameters.

        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point
            img_width, img_height: Image dimensions
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.img_width = img_width
        self.img_height = img_height

        # Intrinsic matrix
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, points_3d, T_camera):
        """
        Project 3D world points to 2D image coordinates.

        Args:
            points_3d: (N, 3) array of 3D points in world frame
            T_camera: (4, 4) camera pose matrix (world to camera transform)

        Returns:
            points_2d: (N, 2) array of 2D image coordinates
            depths: (N,) array of point depths
        """
        # Transform to camera frame
        N = points_3d.shape[0]
        points_h = np.hstack([points_3d, np.ones((N, 1))])  # Homogeneous
        points_cam = (T_camera @ points_h.T).T[:, :3]  # (N, 3)

        # Project to image plane
        depths = points_cam[:, 2]
        x = self.fx * points_cam[:, 0] / depths + self.cx
        y = self.fy * points_cam[:, 1] / depths + self.cy

        points_2d = np.stack([x, y], axis=1)
        return points_2d, depths


def pose_matrix(tx, ty, tz, rx, ry, rz):
    """
    Create 4x4 pose matrix from translation and Euler angles (XYZ order).

    Returns transformation from world to camera frame.
    """
    # Rotation matrices
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    R = Rz @ Ry @ Rx
    t = np.array([tx, ty, tz])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


# ============================================================================
# Image-Based Visual Servoing (IBVS)
# ============================================================================


class IBVS:
    """
    Image-Based Visual Servoing controller.

    Controls camera velocity to minimize image feature error.
    """

    def __init__(self, camera, lambda_gain=0.5):
        """
        Args:
            camera: PinholeCamera instance
            lambda_gain: Control gain (larger = faster but less stable)
        """
        self.camera = camera
        self.lambda_gain = lambda_gain

    def compute_interaction_matrix(self, points_2d, depths):
        """
        Compute the interaction matrix (image Jacobian) for point features.

        For a point feature s = (x, y), the interaction matrix L relates
        the feature velocity to camera velocity: ṡ = L * v

        Args:
            points_2d: (N, 2) image coordinates
            depths: (N,) depths of points

        Returns:
            L: (2N, 6) interaction matrix
        """
        N = points_2d.shape[0]
        L = np.zeros((2 * N, 6))

        fx, fy = self.camera.fx, self.camera.fy
        cx, cy = self.camera.cx, self.camera.cy

        for i in range(N):
            x = (points_2d[i, 0] - cx) / fx  # Normalized image coordinates
            y = (points_2d[i, 1] - cy) / fy
            Z = depths[i]

            # Interaction matrix for point (x, y)
            # [ẋ]   [-1/Z   0   x/Z   xy    -(1+x²)   y ] [vx]
            # [ẏ] = [ 0   -1/Z  y/Z  1+y²    -xy    -x ] [vy]
            #                                            [vz]
            #                                            [ωx]
            #                                            [ωy]
            #                                            [ωz]

            L[2 * i, :] = [-fx / Z, 0, fx * x / Z, fx * x * y, -fx * (1 + x**2), fx * y]
            L[2 * i + 1, :] = [
                0,
                -fy / Z,
                fy * y / Z,
                fy * (1 + y**2),
                -fy * x * y,
                -fy * x,
            ]

        return L

    def compute_control(self, current_features, desired_features, depths):
        """
        Compute camera velocity to minimize feature error.

        Control law: v = -λ * L⁺ * (s - s*)

        Args:
            current_features: (N, 2) current image coordinates
            desired_features: (N, 2) desired image coordinates
            depths: (N,) current depths

        Returns:
            v: (6,) camera velocity [vx, vy, vz, ωx, ωy, ωz]
            error: Feature error norm
        """
        # Feature error
        e = (current_features - desired_features).flatten()  # (2N,)

        # Interaction matrix at current features
        L = self.compute_interaction_matrix(current_features, depths)

        # Pseudoinverse
        L_pinv = np.linalg.pinv(L)

        # Control velocity
        v = -self.lambda_gain * L_pinv @ e

        return v, np.linalg.norm(e)


# ============================================================================
# Simulation
# ============================================================================


def create_target_points(size=0.2):
    """Create 4 corner points of a square target centered at origin."""
    half = size / 2
    return np.array(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]
    )


def update_pose(T, velocity, dt):
    """
    Update camera pose given velocity command.

    Simple first-order integration.
    """
    vx, vy, vz, wx, wy, wz = velocity

    # Extract current rotation and translation
    R = T[:3, :3]
    t = T[:3, 3]

    # Update translation (in camera frame, then transform to world)
    dt_cam = np.array([vx, vy, vz]) * dt
    t_new = t + R.T @ dt_cam

    # Update rotation (small angle approximation)
    dR = np.array(
        [[1, -wz * dt, wy * dt], [wz * dt, 1, -wx * dt], [-wy * dt, wx * dt, 1]]
    )
    R_new = R @ dR

    # Re-orthogonalize
    U, _, Vt = np.linalg.svd(R_new)
    R_new = U @ Vt

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new

    return T_new


def run_ibvs_simulation(visualize=True):
    """
    Run the IBVS simulation.

    Demonstrates visual servoing from an initial camera pose to a desired pose.
    """
    print("=" * 60)
    print("Demo 01: Image-Based Visual Servoing (IBVS)")
    print("=" * 60)
    print()
    print("Scenario: Camera viewing a square target")
    print("Goal: Move camera to achieve desired image of target")
    print()

    # Setup
    camera = PinholeCamera()
    ibvs = IBVS(camera, lambda_gain=0.8)
    target_points = create_target_points(size=0.3)

    # Initial camera pose (offset and rotated from desired)
    T_initial = pose_matrix(
        tx=0.1,
        ty=-0.1,
        tz=0.8,  # Slightly offset
        rx=0.2,
        ry=-0.15,
        rz=0.1,  # Slightly rotated
    )

    # Desired camera pose (directly facing target)
    T_desired = pose_matrix(tx=0.0, ty=0.0, tz=0.6, rx=0.0, ry=0.0, rz=0.0)

    # Compute desired image features
    desired_features, _ = camera.project(target_points, T_desired)

    # Simulation parameters
    dt = 0.05
    max_iterations = 500
    convergence_threshold = 1.0  # pixels

    # History for plotting
    T_current = T_initial.copy()
    feature_history = []
    error_history = []
    pose_history = [T_current.copy()]

    print("Running IBVS control loop...")
    print(
        f"Initial pose: t=[{T_initial[0,3]:.3f}, {T_initial[1,3]:.3f}, {T_initial[2,3]:.3f}]"
    )
    print(
        f"Desired pose: t=[{T_desired[0,3]:.3f}, {T_desired[1,3]:.3f}, {T_desired[2,3]:.3f}]"
    )
    print()

    for i in range(max_iterations):
        # Project points to current image
        current_features, depths = camera.project(target_points, T_current)

        # Compute control
        velocity, error = ibvs.compute_control(
            current_features, desired_features, depths
        )

        # Store history
        feature_history.append(current_features.copy())
        error_history.append(error)

        # Check convergence
        if error < convergence_threshold:
            print(f"Converged at iteration {i} with error {error:.2f} pixels")
            break

        # Update pose
        T_current = update_pose(T_current, velocity, dt)
        pose_history.append(T_current.copy())

        if i % 50 == 0:
            print(f"  Iteration {i}: error = {error:.2f} pixels")

    print()
    print(
        f"Final pose: t=[{T_current[0,3]:.3f}, {T_current[1,3]:.3f}, {T_current[2,3]:.3f}]"
    )

    if visualize:
        visualize_ibvs_results(
            camera,
            target_points,
            desired_features,
            feature_history,
            error_history,
            pose_history,
        )

    return error_history


def visualize_ibvs_results(
    camera,
    target_points,
    desired_features,
    feature_history,
    error_history,
    pose_history,
):
    """Create comprehensive visualization of IBVS results."""

    fig = plt.figure(figsize=(15, 10))

    # 1. Feature trajectories in image space
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim(0, camera.img_width)
    ax1.set_ylim(camera.img_height, 0)  # Image coordinates (y down)
    ax1.set_xlabel("u (pixels)")
    ax1.set_ylabel("v (pixels)")
    ax1.set_title("Feature Trajectories in Image Space")
    ax1.set_aspect("equal")

    colors = ["red", "green", "blue", "orange"]
    for point_idx in range(4):
        # Trajectory
        traj = np.array([f[point_idx] for f in feature_history])
        ax1.plot(
            traj[:, 0], traj[:, 1], "-", color=colors[point_idx], alpha=0.5, linewidth=1
        )

        # Start point
        ax1.plot(
            traj[0, 0],
            traj[0, 1],
            "o",
            color=colors[point_idx],
            markersize=10,
            label=f"Start P{point_idx+1}",
        )

        # End point
        ax1.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[point_idx], markersize=8)

        # Desired point
        ax1.plot(
            desired_features[point_idx, 0],
            desired_features[point_idx, 1],
            "*",
            color=colors[point_idx],
            markersize=15,
        )

    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Error over iterations
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(error_history, "b-", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Feature Error (pixels)")
    ax2.set_title("Convergence of Feature Error")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="r", linestyle="--", label="Convergence threshold")
    ax2.legend()

    # 3. Camera trajectory in 3D
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")

    # Target points
    ax3.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        c="red",
        s=100,
        marker="s",
        label="Target",
    )

    # Connect target points
    for i in range(4):
        j = (i + 1) % 4
        ax3.plot(
            [target_points[i, 0], target_points[j, 0]],
            [target_points[i, 1], target_points[j, 1]],
            [target_points[i, 2], target_points[j, 2]],
            "r-",
            linewidth=2,
        )

    # Camera trajectory
    cam_positions = np.array([np.linalg.inv(T)[:3, 3] for T in pose_history])
    ax3.plot(
        cam_positions[:, 0],
        cam_positions[:, 1],
        cam_positions[:, 2],
        "b-",
        linewidth=2,
        label="Camera path",
    )
    ax3.scatter(
        cam_positions[0, 0],
        cam_positions[0, 1],
        cam_positions[0, 2],
        c="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax3.scatter(
        cam_positions[-1, 0],
        cam_positions[-1, 1],
        cam_positions[-1, 2],
        c="blue",
        s=100,
        marker="^",
        label="End",
    )

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Camera Trajectory in 3D")
    ax3.legend()

    # 4. Initial vs Final image
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim(0, camera.img_width)
    ax4.set_ylim(camera.img_height, 0)

    # Initial features
    initial_features = feature_history[0]
    final_features = feature_history[-1]

    # Draw initial polygon
    init_pts = np.vstack([initial_features, initial_features[0]])
    ax4.plot(init_pts[:, 0], init_pts[:, 1], "r--", linewidth=2, label="Initial")

    # Draw final polygon
    final_pts = np.vstack([final_features, final_features[0]])
    ax4.plot(final_pts[:, 0], final_pts[:, 1], "b-", linewidth=2, label="Final")

    # Draw desired polygon
    des_pts = np.vstack([desired_features, desired_features[0]])
    ax4.plot(des_pts[:, 0], des_pts[:, 1], "g:", linewidth=3, label="Desired")

    ax4.set_xlabel("u (pixels)")
    ax4.set_ylabel("v (pixels)")
    ax4.set_title("Initial vs Final Image")
    ax4.legend()
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ibvs_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'ibvs_results.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Run simulation
    error_history = run_ibvs_simulation(visualize=True)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. IBVS controls the camera directly in image feature space
2. The interaction matrix (image Jacobian) maps camera velocity to feature velocity
3. The control law v = -λ L⁺ e minimizes feature error exponentially
4. Feature trajectories in image space are generally NOT straight lines
5. IBVS is robust to calibration errors but can have local minima

This classical approach was dominant in robotics before deep learning.
Modern approaches (Lecture 13) learn policies end-to-end from pixels.
    """)

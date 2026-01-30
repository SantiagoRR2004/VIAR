"""
Demo 02: Extended Kalman Filter SLAM (EKF-SLAM)
===============================================

This demo implements a simplified 2D EKF-SLAM system where a robot
navigates through an environment with point landmarks.

Key Concepts:
- State vector contains robot pose AND landmark positions
- Covariance matrix captures uncertainty and correlations
- As landmarks are observed, uncertainty decreases
- Loop closure dramatically reduces global uncertainty

The joint state vector is:
    x = [x_r, y_r, θ_r, x_l1, y_l1, x_l2, y_l2, ...]

Historical Context:
- EKF-SLAM emerged in the late 1980s-1990s
- Foundational work by Smith, Self, and Cheeseman (1990)
- Limited by O(n²) complexity in number of landmarks
- Replaced by particle filters (FastSLAM) and graph-based methods

Reference: Durrant-Whyte & Bailey, "SLAM Tutorial" (2006)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# ============================================================================
# EKF-SLAM Implementation
# ============================================================================


class EKFSLAM:
    """
    Extended Kalman Filter for Simultaneous Localization and Mapping.

    State vector: [x_robot, y_robot, theta_robot, x_lm1, y_lm1, ...]
    """

    def __init__(self, motion_noise, observation_noise):
        """
        Args:
            motion_noise: (3,) std dev for [x, y, theta] motion
            observation_noise: (2,) std dev for [range, bearing] observations
        """
        # Noise parameters
        self.motion_noise = np.array(motion_noise)
        self.observation_noise = np.array(observation_noise)

        # State: robot pose only initially
        self.state = np.zeros(3)  # [x, y, theta]
        self.covariance = np.diag([0.01, 0.01, 0.01])

        # Landmark management
        self.n_landmarks = 0
        self.landmark_ids = {}  # Map from landmark ID to index in state

    @property
    def robot_pose(self):
        """Get robot [x, y, theta]."""
        return self.state[:3]

    @property
    def landmarks(self):
        """Get landmark positions as (n, 2) array."""
        if self.n_landmarks == 0:
            return np.array([]).reshape(0, 2)
        return self.state[3:].reshape(-1, 2)

    def predict(self, control):
        """
        Prediction step: propagate robot state with motion model.

        Args:
            control: [delta_x, delta_y, delta_theta] in robot frame
        """
        dx, dy, dtheta = control
        theta = self.state[2]

        # Motion model (velocity motion model)
        # Transform control from robot frame to world frame
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Update robot pose
        self.state[0] += dx * cos_t - dy * sin_t
        self.state[1] += dx * sin_t + dy * cos_t
        self.state[2] += dtheta
        self.state[2] = self._wrap_angle(self.state[2])

        # Jacobian of motion model w.r.t. state
        n = len(self.state)
        F = np.eye(n)
        F[0, 2] = -dx * sin_t - dy * cos_t
        F[1, 2] = dx * cos_t - dy * sin_t

        # Process noise (only affects robot pose)
        Q = np.zeros((n, n))
        Q[:3, :3] = np.diag(self.motion_noise**2)

        # Update covariance
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, landmark_id, observation):
        """
        Update step: incorporate landmark observation.

        Args:
            landmark_id: Unique identifier for the landmark
            observation: [range, bearing] measurement
        """
        z_range, z_bearing = observation

        if landmark_id not in self.landmark_ids:
            # New landmark: initialize
            self._add_landmark(landmark_id, z_range, z_bearing)
        else:
            # Existing landmark: update
            self._update_landmark(landmark_id, z_range, z_bearing)

    def _add_landmark(self, landmark_id, z_range, z_bearing):
        """Initialize a new landmark from first observation."""
        # Compute landmark position in world frame
        x, y, theta = self.state[:3]
        lm_x = x + z_range * np.cos(theta + z_bearing)
        lm_y = y + z_range * np.sin(theta + z_bearing)

        # Extend state vector
        self.state = np.append(self.state, [lm_x, lm_y])
        self.landmark_ids[landmark_id] = self.n_landmarks
        self.n_landmarks += 1

        # Extend covariance matrix
        # New landmark has high initial uncertainty
        n_old = len(self.covariance)
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.covariance

        # Initialize landmark covariance based on observation uncertainty
        r_var = self.observation_noise[0] ** 2
        b_var = self.observation_noise[1] ** 2

        cos_tb = np.cos(theta + z_bearing)
        sin_tb = np.sin(theta + z_bearing)

        # Jacobian of landmark init w.r.t. [x, y, theta, range, bearing]
        # Simplified: use high initial uncertainty
        P_new[n_old, n_old] = r_var * cos_tb**2 + z_range**2 * b_var * sin_tb**2 + 10.0
        P_new[n_old + 1, n_old + 1] = (
            r_var * sin_tb**2 + z_range**2 * b_var * cos_tb**2 + 10.0
        )

        self.covariance = P_new

    def _update_landmark(self, landmark_id, z_range, z_bearing):
        """Update state using observation of known landmark."""
        lm_idx = self.landmark_ids[landmark_id]
        state_idx = 3 + 2 * lm_idx  # Index in state vector

        # Get current estimates
        x, y, theta = self.state[:3]
        lm_x, lm_y = self.state[state_idx : state_idx + 2]

        # Predicted observation
        dx = lm_x - x
        dy = lm_y - y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        z_pred_range = sqrt_q
        z_pred_bearing = self._wrap_angle(np.arctan2(dy, dx) - theta)

        # Innovation
        z = np.array([z_range, z_bearing])
        z_pred = np.array([z_pred_range, z_pred_bearing])
        innovation = z - z_pred
        innovation[1] = self._wrap_angle(innovation[1])

        # Jacobian of observation model
        n = len(self.state)
        H = np.zeros((2, n))

        # w.r.t robot pose
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1

        # w.r.t landmark position
        H[0, state_idx] = dx / sqrt_q
        H[0, state_idx + 1] = dy / sqrt_q
        H[1, state_idx] = -dy / q
        H[1, state_idx + 1] = dx / q

        # Observation noise
        R = np.diag(self.observation_noise**2)

        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ innovation
        self.state[2] = self._wrap_angle(self.state[2])

        I = np.eye(n)
        self.covariance = (I - K @ H) @ self.covariance

        # Ensure symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_ellipse_params(self, index, n_std=2):
        """
        Get ellipse parameters for visualizing uncertainty.

        Args:
            index: 0 for robot, 1+ for landmarks
            n_std: Number of standard deviations

        Returns:
            (center, width, height, angle) for matplotlib Ellipse
        """
        if index == 0:
            # Robot position uncertainty
            mean = self.state[:2]
            cov = self.covariance[:2, :2]
        else:
            # Landmark uncertainty
            idx = 3 + 2 * (index - 1)
            mean = self.state[idx : idx + 2]
            cov = self.covariance[idx : idx + 2, idx : idx + 2]

        # Eigendecomposition for ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])

        return mean, width, height, angle


# ============================================================================
# Simulation Environment
# ============================================================================


class SLAMSimulator:
    """
    Simulates robot motion and landmark observations for SLAM.
    """

    def __init__(
        self,
        landmarks,
        motion_noise_std,
        observation_noise_std,
        max_range=5.0,
        fov=np.pi,
    ):
        """
        Args:
            landmarks: (N, 2) true landmark positions
            motion_noise_std: (3,) motion noise [x, y, theta]
            observation_noise_std: (2,) observation noise [range, bearing]
            max_range: Maximum observation range
            fov: Field of view (radians, centered on heading)
        """
        self.landmarks = np.array(landmarks)
        self.motion_noise_std = np.array(motion_noise_std)
        self.observation_noise_std = np.array(observation_noise_std)
        self.max_range = max_range
        self.fov = fov

        # True robot state
        self.true_pose = np.array([0.0, 0.0, 0.0])

    def move(self, control):
        """
        Move robot and return noisy odometry.

        Args:
            control: [dx, dy, dtheta] commanded motion in robot frame

        Returns:
            noisy_control: Motion with added noise
        """
        dx, dy, dtheta = control
        theta = self.true_pose[2]

        # Apply true motion
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        self.true_pose[0] += dx * cos_t - dy * sin_t
        self.true_pose[1] += dx * sin_t + dy * cos_t
        self.true_pose[2] += dtheta

        # Return noisy odometry
        noise = np.random.randn(3) * self.motion_noise_std
        return control + noise

    def observe(self):
        """
        Get observations of visible landmarks.

        Returns:
            observations: List of (landmark_id, [range, bearing])
        """
        observations = []
        x, y, theta = self.true_pose

        for i, (lm_x, lm_y) in enumerate(self.landmarks):
            # Compute true range and bearing
            dx = lm_x - x
            dy = lm_y - y
            true_range = np.sqrt(dx**2 + dy**2)
            true_bearing = np.arctan2(dy, dx) - theta

            # Wrap bearing to [-pi, pi]
            true_bearing = (true_bearing + np.pi) % (2 * np.pi) - np.pi

            # Check visibility
            if true_range <= self.max_range and abs(true_bearing) <= self.fov / 2:
                # Add noise
                noise = np.random.randn(2) * self.observation_noise_std
                noisy_obs = np.array([true_range, true_bearing]) + noise
                observations.append((i, noisy_obs))

        return observations


# ============================================================================
# Visualization
# ============================================================================


def visualize_slam(
    slam, simulator, trajectory_true, trajectory_est, observations_history
):
    """Create comprehensive SLAM visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Final map with uncertainty
    ax1 = axes[0, 0]
    ax1.set_title("EKF-SLAM Result with Uncertainty Ellipses", fontsize=12)

    # True landmarks
    ax1.scatter(
        simulator.landmarks[:, 0],
        simulator.landmarks[:, 1],
        c="red",
        s=150,
        marker="*",
        label="True landmarks",
        zorder=5,
    )

    # Estimated landmarks with uncertainty
    for i in range(slam.n_landmarks):
        center, width, height, angle = slam.get_ellipse_params(i + 1, n_std=2)
        ellipse = Ellipse(
            center, width, height, angle=angle, fill=False, color="blue", linewidth=2
        )
        ax1.add_patch(ellipse)
        ax1.scatter(center[0], center[1], c="blue", s=80, marker="s", zorder=4)

    # Trajectories
    traj_true = np.array(trajectory_true)
    traj_est = np.array(trajectory_est)
    ax1.plot(
        traj_true[:, 0],
        traj_true[:, 1],
        "g-",
        linewidth=2,
        label="True trajectory",
        alpha=0.7,
    )
    ax1.plot(
        traj_est[:, 0],
        traj_est[:, 1],
        "b--",
        linewidth=2,
        label="Estimated trajectory",
        alpha=0.7,
    )

    # Robot uncertainty ellipse
    center, width, height, angle = slam.get_ellipse_params(0, n_std=2)
    robot_ellipse = Ellipse(
        center, width, height, angle=angle, fill=False, color="green", linewidth=2
    )
    ax1.add_patch(robot_ellipse)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend(loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # 2. Trajectory error over time
    ax2 = axes[0, 1]
    errors = np.sqrt(np.sum((traj_true - traj_est) ** 2, axis=1))
    ax2.plot(errors, "b-", linewidth=2)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Robot Position Error Over Time")
    ax2.grid(True, alpha=0.3)

    # 3. Covariance trace over time
    ax3 = axes[1, 0]
    # We'll need to track this during simulation - for now show final covariance structure
    cov_img = ax3.imshow(slam.covariance, cmap="viridis", aspect="auto")
    plt.colorbar(cov_img, ax=ax3)
    ax3.set_title("Final Covariance Matrix Structure")
    ax3.set_xlabel("State index")
    ax3.set_ylabel("State index")

    # Add labels
    n_lm = slam.n_landmarks
    tick_labels = ["x", "y", "θ"] + [
        f"L{i}x" if j == 0 else f"L{i}y" for i in range(n_lm) for j in range(2)
    ]
    ax3.set_xticks(range(len(tick_labels)))
    ax3.set_yticks(range(len(tick_labels)))
    ax3.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax3.set_yticklabels(tick_labels, fontsize=8)

    # 4. Landmark estimation error
    ax4 = axes[1, 1]
    if slam.n_landmarks > 0:
        est_landmarks = slam.landmarks
        true_landmarks = simulator.landmarks

        # Match estimated to true landmarks by ID
        lm_errors = []
        for lm_id, idx in slam.landmark_ids.items():
            est = est_landmarks[idx]
            true = true_landmarks[lm_id]
            lm_errors.append(np.linalg.norm(est - true))

        ax4.bar(range(len(lm_errors)), lm_errors, color="blue", alpha=0.7)
        ax4.set_xlabel("Landmark ID")
        ax4.set_ylabel("Position error (m)")
        ax4.set_title("Landmark Position Errors")
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("ekf_slam_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'ekf_slam_results.png'")


def run_slam_demo():
    """Run the EKF-SLAM demonstration."""

    print("=" * 60)
    print("Demo 02: Extended Kalman Filter SLAM (EKF-SLAM)")
    print("=" * 60)
    print()

    # Define environment
    np.random.seed(42)

    # Landmarks arranged in a pattern
    landmarks = np.array(
        [
            [3, 2],
            [6, 2],
            [9, 3],
            [2, 5],
            [5, 5],
            [8, 6],
            [3, 8],
            [6, 9],
            [9, 8],
            [1, 1],
            [10, 10],
        ]
    )

    print(f"Environment: {len(landmarks)} landmarks")
    print(f"Landmarks at: {landmarks[:3].tolist()} ...")
    print()

    # Noise parameters
    motion_noise = [0.1, 0.1, 0.05]  # [x, y, theta] std dev
    obs_noise = [0.2, 0.05]  # [range, bearing] std dev

    # Create simulator and SLAM
    simulator = SLAMSimulator(
        landmarks=landmarks,
        motion_noise_std=motion_noise,
        observation_noise_std=obs_noise,
        max_range=4.0,
        fov=np.pi * 1.2,  # 216 degrees FOV
    )

    slam = EKFSLAM(motion_noise=motion_noise, observation_noise=obs_noise)

    # Define trajectory (rectangular path)
    trajectory_commands = []

    # Move right
    for _ in range(20):
        trajectory_commands.append([0.5, 0, 0])
    # Turn left
    trajectory_commands.append([0, 0, np.pi / 2])
    # Move up
    for _ in range(15):
        trajectory_commands.append([0.5, 0, 0])
    # Turn left
    trajectory_commands.append([0, 0, np.pi / 2])
    # Move left
    for _ in range(20):
        trajectory_commands.append([0.5, 0, 0])
    # Turn left
    trajectory_commands.append([0, 0, np.pi / 2])
    # Move down (loop closure)
    for _ in range(15):
        trajectory_commands.append([0.5, 0, 0])

    print(f"Trajectory: {len(trajectory_commands)} motion commands")
    print()

    # Run simulation
    trajectory_true = [simulator.true_pose.copy()]
    trajectory_est = [slam.robot_pose.copy()]
    observations_history = []

    print("Running EKF-SLAM...")
    for i, cmd in enumerate(trajectory_commands):
        # Move robot and get noisy odometry
        noisy_odom = simulator.move(cmd)

        # Prediction step
        slam.predict(noisy_odom)

        # Get observations
        observations = simulator.observe()
        observations_history.append(observations)

        # Update step for each observation
        for lm_id, obs in observations:
            slam.update(lm_id, obs)

        # Record trajectory
        trajectory_true.append(simulator.true_pose.copy())
        trajectory_est.append(slam.robot_pose.copy())

        if i % 20 == 0:
            print(
                f"  Step {i}: {slam.n_landmarks} landmarks mapped, "
                f"robot at ({slam.robot_pose[0]:.2f}, {slam.robot_pose[1]:.2f})"
            )

    print()
    print(f"Final: {slam.n_landmarks} landmarks discovered")

    # Calculate final errors
    final_pos_error = np.linalg.norm(simulator.true_pose[:2] - slam.robot_pose[:2])
    print(f"Final robot position error: {final_pos_error:.3f} m")

    if slam.n_landmarks > 0:
        landmark_errors = []
        for lm_id, idx in slam.landmark_ids.items():
            est = slam.landmarks[idx]
            true = simulator.landmarks[lm_id]
            landmark_errors.append(np.linalg.norm(est - true))
        print(f"Mean landmark error: {np.mean(landmark_errors):.3f} m")

    # Visualize
    visualize_slam(
        slam, simulator, trajectory_true, trajectory_est, observations_history
    )

    return slam


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    slam = run_slam_demo()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. EKF-SLAM maintains a joint state of robot pose and all landmarks
2. The covariance matrix captures uncertainty AND correlations
3. Observing a landmark improves estimates of ALL correlated states
4. Loop closure (revisiting landmarks) dramatically reduces uncertainty
5. Computational complexity is O(n²) in number of landmarks

Limitations that led to newer methods:
- Cannot represent multi-modal distributions (particle filters can)
- Linearization errors accumulate over time
- Scales poorly to large environments

Modern approaches use:
- Particle filters (FastSLAM)
- Graph-based optimization (g2o, GTSAM)
- Deep learning (Neural SLAM - Lecture 12 Part IV)
    """)

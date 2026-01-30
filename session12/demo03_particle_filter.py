"""
Demo 03: Particle Filter Localization (Monte Carlo Localization)
=================================================================

This demo implements a particle filter for robot localization,
a key classical method that preceded neural approaches to SLAM.

Key Concepts:
- Represent belief as a set of weighted particles
- Each particle is a hypothesis about robot pose
- Resampling focuses particles on likely hypotheses
- Can represent multi-modal distributions (unlike EKF)

The particle filter algorithm:
1. Predict: Move particles according to motion model
2. Update: Weight particles by observation likelihood
3. Resample: Duplicate high-weight, remove low-weight particles

Historical Context:
- Particle filters emerged in the 1990s for non-linear/non-Gaussian problems
- Monte Carlo Localization (MCL) applied them to robotics (1999)
- FastSLAM (2002) combined particle filters with EKF for landmarks
- Key advantage: Can handle kidnapped robot problem (global localization)

Reference: Thrun et al., "Probabilistic Robotics" (2005)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
import matplotlib.animation as animation
from tqdm import tqdm
import cv2

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# Environment: Simple 2D World with Landmarks
# ============================================================================


class LandmarkWorld:
    """
    Simple 2D world with point landmarks for localization.

    The robot can observe range and bearing to landmarks within sensor range.
    """

    def __init__(self, world_size=20, num_landmarks=8):
        self.world_size = world_size

        # Generate random landmarks
        self.landmarks = np.random.uniform(1, world_size - 1, (num_landmarks, 2))

        # Sensor parameters
        self.sensor_range = 8.0
        self.sensor_fov = np.pi  # 180 degrees

        # Noise parameters
        self.range_noise_std = 0.3
        self.bearing_noise_std = 0.1

    def get_observations(self, robot_pose):
        """
        Get observations of landmarks from robot pose.

        Returns list of (landmark_id, range, bearing) for visible landmarks.
        """
        x, y, theta = robot_pose
        observations = []

        for i, (lx, ly) in enumerate(self.landmarks):
            # Compute range and bearing
            dx = lx - x
            dy = ly - y
            true_range = np.sqrt(dx**2 + dy**2)
            true_bearing = np.arctan2(dy, dx) - theta

            # Normalize bearing to [-pi, pi]
            true_bearing = (true_bearing + np.pi) % (2 * np.pi) - np.pi

            # Check if within sensor range and FOV
            if (
                true_range <= self.sensor_range
                and abs(true_bearing) <= self.sensor_fov / 2
            ):
                # Add noise
                noisy_range = true_range + np.random.randn() * self.range_noise_std
                noisy_bearing = (
                    true_bearing + np.random.randn() * self.bearing_noise_std
                )

                observations.append((i, noisy_range, noisy_bearing))

        return observations


# ============================================================================
# Particle Filter Implementation
# ============================================================================


class ParticleFilter:
    """
    Particle Filter for robot localization.

    Each particle represents a hypothesis about the robot's pose (x, y, theta).
    """

    def __init__(self, num_particles, world_size, motion_noise, observation_model):
        """
        Args:
            num_particles: Number of particles
            world_size: Size of the world for initialization
            motion_noise: (3,) noise std for [x, y, theta]
            observation_model: Function to compute observation likelihood
        """
        self.num_particles = num_particles
        self.world_size = world_size
        self.motion_noise = np.array(motion_noise)
        self.observation_model = observation_model

        # Initialize particles uniformly
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.uniform(0, world_size, num_particles)
        self.particles[:, 1] = np.random.uniform(0, world_size, num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

        # Uniform weights initially
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control):
        """
        Prediction step: Move particles according to motion model.

        Args:
            control: [dx, dy, dtheta] motion command in robot frame
        """
        dx, dy, dtheta = control

        for i in range(self.num_particles):
            # Add noise to control
            noisy_dx = dx + np.random.randn() * self.motion_noise[0]
            noisy_dy = dy + np.random.randn() * self.motion_noise[1]
            noisy_dtheta = dtheta + np.random.randn() * self.motion_noise[2]

            # Apply motion in world frame
            theta = self.particles[i, 2]
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            self.particles[i, 0] += noisy_dx * cos_t - noisy_dy * sin_t
            self.particles[i, 1] += noisy_dx * sin_t + noisy_dy * cos_t
            self.particles[i, 2] += noisy_dtheta

            # Wrap angle
            self.particles[i, 2] = (self.particles[i, 2] + np.pi) % (2 * np.pi) - np.pi

            # Keep in bounds
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0, self.world_size)
            self.particles[i, 1] = np.clip(self.particles[i, 1], 0, self.world_size)

    def update(self, observations, landmarks):
        """
        Update step: Reweight particles based on observations.

        Args:
            observations: List of (landmark_id, range, bearing)
            landmarks: Array of landmark positions
        """
        if len(observations) == 0:
            return

        for i in range(self.num_particles):
            particle = self.particles[i]
            likelihood = 1.0

            for lm_id, obs_range, obs_bearing in observations:
                # Compute expected observation from particle
                lx, ly = landmarks[lm_id]
                dx = lx - particle[0]
                dy = ly - particle[1]
                expected_range = np.sqrt(dx**2 + dy**2)
                expected_bearing = np.arctan2(dy, dx) - particle[2]
                expected_bearing = (expected_bearing + np.pi) % (2 * np.pi) - np.pi

                # Compute likelihood (Gaussian)
                range_prob = np.exp(-0.5 * ((obs_range - expected_range) / 0.5) ** 2)
                bearing_prob = np.exp(
                    -0.5 * ((obs_bearing - expected_bearing) / 0.2) ** 2
                )

                likelihood *= range_prob * bearing_prob

            self.weights[i] = likelihood

        # Normalize weights
        self.weights = self.weights / (np.sum(self.weights) + 1e-10)

    def resample(self):
        """
        Resample particles according to weights.

        Uses systematic resampling for lower variance.
        """
        # Compute cumulative sum
        cumsum = np.cumsum(self.weights)

        # Systematic resampling
        positions = (
            np.arange(self.num_particles) + np.random.uniform()
        ) / self.num_particles

        new_particles = np.zeros_like(self.particles)
        idx = 0
        for i, pos in enumerate(positions):
            while cumsum[idx] < pos:
                idx += 1
            new_particles[i] = self.particles[idx]

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        """Get weighted mean estimate of robot pose."""
        mean_x = np.sum(self.weights * self.particles[:, 0])
        mean_y = np.sum(self.weights * self.particles[:, 1])

        # For angle, use circular mean
        mean_cos = np.sum(self.weights * np.cos(self.particles[:, 2]))
        mean_sin = np.sum(self.weights * np.sin(self.particles[:, 2]))
        mean_theta = np.arctan2(mean_sin, mean_cos)

        return np.array([mean_x, mean_y, mean_theta])

    def get_effective_particles(self):
        """Compute effective number of particles (measure of degeneracy)."""
        return 1.0 / np.sum(self.weights**2)


# ============================================================================
# Simulation
# ============================================================================


def run_particle_filter_demo():
    """Run the particle filter localization demo."""

    print("=" * 60)
    print("Demo 03: Particle Filter Localization")
    print("=" * 60)
    print()

    # Setup
    world = LandmarkWorld(world_size=20, num_landmarks=10)

    # True robot pose
    true_pose = np.array([3.0, 3.0, 0.0])

    # Motion noise for particle filter
    motion_noise = [0.2, 0.2, 0.1]

    # Create particle filter
    num_particles = 500
    pf = ParticleFilter(
        num_particles=num_particles,
        world_size=world.world_size,
        motion_noise=motion_noise,
        observation_model=None,  # We handle this in update
    )

    print(f"World size: {world.world_size} x {world.world_size}")
    print(f"Number of landmarks: {len(world.landmarks)}")
    print(f"Number of particles: {num_particles}")
    print(
        f"True initial pose: [{true_pose[0]:.1f}, {true_pose[1]:.1f}, {np.degrees(true_pose[2]):.1f}°]"
    )
    print()

    # Define trajectory (square path)
    controls = []

    # Move right
    for _ in range(10):
        controls.append([0.5, 0, 0])
    # Turn left 90 degrees
    for _ in range(3):
        controls.append([0, 0, np.pi / 6])
    # Move up
    for _ in range(10):
        controls.append([0.5, 0, 0])
    # Turn left 90 degrees
    for _ in range(3):
        controls.append([0, 0, np.pi / 6])
    # Move left
    for _ in range(10):
        controls.append([0.5, 0, 0])
    # Turn left 90 degrees
    for _ in range(3):
        controls.append([0, 0, np.pi / 6])
    # Move down (back to start area)
    for _ in range(10):
        controls.append([0.5, 0, 0])

    # Run simulation
    true_trajectory = [true_pose.copy()]
    estimated_trajectory = [pf.get_estimate()]
    particle_history = [pf.particles.copy()]
    n_eff_history = [pf.get_effective_particles()]

    print("Running particle filter...")

    for i, control in enumerate(tqdm(controls, desc="Localizing")):
        # Move true robot (with some noise)
        dx, dy, dtheta = control
        theta = true_pose[2]
        true_pose[0] += (
            dx * np.cos(theta) - dy * np.sin(theta) + np.random.randn() * 0.05
        )
        true_pose[1] += (
            dx * np.sin(theta) + dy * np.cos(theta) + np.random.randn() * 0.05
        )
        true_pose[2] += dtheta + np.random.randn() * 0.02
        true_pose[2] = (true_pose[2] + np.pi) % (2 * np.pi) - np.pi

        # Keep in bounds
        true_pose[0] = np.clip(true_pose[0], 0, world.world_size)
        true_pose[1] = np.clip(true_pose[1], 0, world.world_size)

        # Particle filter predict
        pf.predict(control)

        # Get observations
        observations = world.get_observations(true_pose)

        # Particle filter update
        pf.update(observations, world.landmarks)

        # Resample if effective particles too low
        n_eff = pf.get_effective_particles()
        if n_eff < num_particles / 2:
            pf.resample()

        # Record
        true_trajectory.append(true_pose.copy())
        estimated_trajectory.append(pf.get_estimate())
        particle_history.append(pf.particles.copy())
        n_eff_history.append(pf.get_effective_particles())

    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)

    # Calculate errors
    position_errors = np.sqrt(
        np.sum((true_trajectory[:, :2] - estimated_trajectory[:, :2]) ** 2, axis=1)
    )

    print()
    print(
        f"Final true pose: [{true_pose[0]:.2f}, {true_pose[1]:.2f}, {np.degrees(true_pose[2]):.1f}°]"
    )
    print(
        f"Final estimate: [{estimated_trajectory[-1, 0]:.2f}, {estimated_trajectory[-1, 1]:.2f}, {np.degrees(estimated_trajectory[-1, 2]):.1f}°]"
    )
    print(f"Mean position error: {np.mean(position_errors):.3f} m")
    print(f"Final position error: {position_errors[-1]:.3f} m")

    # Visualize
    visualize_particle_filter(
        world,
        true_trajectory,
        estimated_trajectory,
        particle_history,
        position_errors,
        n_eff_history,
    )

    return pf


def visualize_particle_filter(
    world, true_traj, est_traj, particle_history, errors, n_eff_history
):
    """Create comprehensive visualization of particle filter results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Final state with particles
    ax1 = axes[0, 0]
    ax1.set_xlim(-1, world.world_size + 1)
    ax1.set_ylim(-1, world.world_size + 1)
    ax1.set_aspect("equal")
    ax1.set_title("Particle Filter Localization (Final State)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Landmarks
    ax1.scatter(
        world.landmarks[:, 0],
        world.landmarks[:, 1],
        c="red",
        s=150,
        marker="*",
        label="Landmarks",
        zorder=5,
    )

    # Final particles
    final_particles = particle_history[-1]
    ax1.scatter(
        final_particles[:, 0],
        final_particles[:, 1],
        c="blue",
        s=5,
        alpha=0.3,
        label="Particles",
    )

    # Trajectories
    ax1.plot(
        true_traj[:, 0],
        true_traj[:, 1],
        "g-",
        linewidth=2,
        label="True trajectory",
        zorder=4,
    )
    ax1.plot(
        est_traj[:, 0],
        est_traj[:, 1],
        "b--",
        linewidth=2,
        label="Estimated trajectory",
        zorder=4,
    )

    # Current positions
    ax1.scatter(
        true_traj[-1, 0],
        true_traj[-1, 1],
        c="green",
        s=200,
        marker="o",
        edgecolors="black",
        zorder=6,
        label="True pose",
    )
    ax1.scatter(
        est_traj[-1, 0],
        est_traj[-1, 1],
        c="blue",
        s=200,
        marker="^",
        edgecolors="black",
        zorder=6,
        label="Estimated pose",
    )

    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. Position error over time
    ax2 = axes[0, 1]
    ax2.plot(errors, "b-", linewidth=2)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Localization Error Over Time")
    ax2.grid(True, alpha=0.3)

    # 3. Effective number of particles
    ax3 = axes[1, 0]
    ax3.plot(n_eff_history, "orange", linewidth=2)
    ax3.axhline(
        y=len(particle_history[0]) / 2,
        color="r",
        linestyle="--",
        label="Resample threshold",
    )
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Effective particles")
    ax3.set_title("Effective Number of Particles (Neff)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Particle evolution (snapshots)
    ax4 = axes[1, 1]
    ax4.set_xlim(-1, world.world_size + 1)
    ax4.set_ylim(-1, world.world_size + 1)
    ax4.set_aspect("equal")
    ax4.set_title("Particle Cloud Evolution")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")

    # Landmarks
    ax4.scatter(
        world.landmarks[:, 0],
        world.landmarks[:, 1],
        c="red",
        s=100,
        marker="*",
        zorder=5,
    )

    # Show particles at different times
    times = [0, len(particle_history) // 4, len(particle_history) // 2, -1]
    colors = ["lightblue", "skyblue", "dodgerblue", "darkblue"]
    labels = ["Start", "25%", "50%", "End"]

    for t, color, label in zip(times, colors, labels):
        particles = particle_history[t]
        ax4.scatter(
            particles[:, 0], particles[:, 1], c=color, s=3, alpha=0.5, label=label
        )

    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("particle_filter_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'particle_filter_results.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    pf = run_particle_filter_demo()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. Particles represent belief as a set of weighted samples
2. Can represent multi-modal distributions (unlike EKF)
3. Prediction: Move particles with motion model + noise
4. Update: Reweight by observation likelihood
5. Resample: Focus computational resources on likely hypotheses

Advantages over EKF-SLAM:
- No Gaussian assumption
- Can handle global localization (kidnapped robot)
- Better with non-linear models

Disadvantages:
- Scales poorly with state dimension
- Particle degeneracy requires careful resampling
- More computationally intensive

This led to:
- FastSLAM: Particle filter for poses, EKF for landmarks
- Modern neural approaches that learn observation models
    """)

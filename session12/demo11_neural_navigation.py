"""
Demo 11: Neural Navigation - End-to-End Visual Navigation
==========================================================

This demo shows how neural networks can learn navigation policies
directly from visual observations, without explicit mapping or planning.

Key Concepts:
- CNN processes egocentric visual input
- LSTM captures spatial memory
- Policy outputs navigation actions
- Point-goal navigation: reach (dx, dy) relative goal

This is a simplified version of the approach used in:
- Neural SLAM (2020)
- Habitat Challenge methods
- AI2-THOR navigation

The demo shows the transition from explicit SLAM to learned navigation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings("ignore")

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Navigation Environment
# ============================================================================


class VisualNavigationEnv:
    """
    Grid-based navigation environment with visual observations.

    - Agent sees egocentric RGB view (what's in front)
    - Actions: turn left, turn right, move forward
    - Goal: reach target location
    - Observations include visual + goal direction

    This is similar to environments used in Habitat/AI2-THOR
    but simplified for educational purposes.
    """

    def __init__(self, grid_size=15, img_size=64, fov=90, max_steps=100):
        """
        Args:
            grid_size: Size of the grid world
            img_size: Size of egocentric observation
            fov: Field of view in degrees
            max_steps: Maximum episode length
        """
        self.grid_size = grid_size
        self.img_size = img_size
        self.fov = np.radians(fov)
        self.max_steps = max_steps

        self.action_dim = 3  # turn left, turn right, forward

        # Colors
        self.colors = {
            "floor": [200, 200, 200],
            "wall": [50, 50, 50],
            "goal": [0, 255, 0],
            "obstacle": [150, 75, 0],
            "agent": [0, 0, 255],
        }

        self.reset()

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0

        # Create grid world (0=floor, 1=wall, 2=obstacle)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Add walls around boundary
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Add some internal obstacles
        n_obstacles = np.random.randint(3, 8)
        for _ in range(n_obstacles):
            ox = np.random.randint(2, self.grid_size - 2)
            oy = np.random.randint(2, self.grid_size - 2)
            self.grid[oy, ox] = 2

        # Place agent
        while True:
            self.agent_pos = np.array(
                [
                    np.random.uniform(2, self.grid_size - 2),
                    np.random.uniform(2, self.grid_size - 2),
                ]
            )
            if self._is_valid_position(self.agent_pos):
                break

        self.agent_angle = np.random.uniform(0, 2 * np.pi)

        # Place goal (at least some distance from agent)
        while True:
            self.goal_pos = np.array(
                [
                    np.random.uniform(2, self.grid_size - 2),
                    np.random.uniform(2, self.grid_size - 2),
                ]
            )
            dist = np.linalg.norm(self.goal_pos - self.agent_pos)
            if self._is_valid_position(self.goal_pos) and dist > 3:
                break

        return self._get_observation(), {}

    def step(self, action):
        """Take an action."""
        self.steps += 1

        if action == 0:  # Turn left
            self.agent_angle += np.pi / 6  # 30 degrees
        elif action == 1:  # Turn right
            self.agent_angle -= np.pi / 6
        elif action == 2:  # Move forward
            new_pos = self.agent_pos + 0.5 * np.array(
                [np.cos(self.agent_angle), np.sin(self.agent_angle)]
            )
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos

        # Normalize angle
        self.agent_angle = self.agent_angle % (2 * np.pi)

        # Check goal reached
        dist_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)

        if dist_to_goal < 0.8:
            reward = 10.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0
            done = True
        else:
            # Reward for getting closer (shaped)
            prev_dist = getattr(self, "_prev_dist", dist_to_goal)
            reward = (prev_dist - dist_to_goal) * 0.5 - 0.01  # Progress - step cost
            self._prev_dist = dist_to_goal
            done = False

        return self._get_observation(), reward, done, False, {"distance": dist_to_goal}

    def _is_valid_position(self, pos):
        """Check if position is valid (not in wall/obstacle)."""
        gx, gy = int(pos[0]), int(pos[1])
        if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
            return False
        return self.grid[gy, gx] == 0

    def _get_observation(self):
        """
        Get egocentric visual observation + goal direction.

        Returns dict with:
        - 'image': Egocentric RGB view
        - 'goal_direction': [dx, dy] relative to agent
        """
        # Render egocentric view
        image = self._render_egocentric()

        # Compute goal direction relative to agent
        goal_vec = self.goal_pos - self.agent_pos

        # Rotate to agent's frame
        cos_a, sin_a = np.cos(-self.agent_angle), np.sin(-self.agent_angle)
        goal_local = np.array(
            [
                goal_vec[0] * cos_a - goal_vec[1] * sin_a,
                goal_vec[0] * sin_a + goal_vec[1] * cos_a,
            ]
        )

        # Normalize by max possible distance
        goal_local = goal_local / self.grid_size

        return {"image": image, "goal_direction": goal_local.astype(np.float32)}

    def _render_egocentric(self):
        """Render first-person view."""
        img = np.full(
            (self.img_size, self.img_size, 3), self.colors["floor"], dtype=np.uint8
        )

        # Simple raycasting for walls
        n_rays = self.img_size
        ray_angles = np.linspace(-self.fov / 2, self.fov / 2, n_rays)

        for i, ray_angle in enumerate(ray_angles):
            angle = self.agent_angle + ray_angle

            # Cast ray
            hit_dist, hit_type = self._cast_ray(self.agent_pos, angle, max_dist=10)

            if hit_dist > 0:
                # Calculate wall height (perspective)
                wall_height = min(int(self.img_size / (hit_dist + 0.1)), self.img_size)

                # Determine color based on what was hit
                if hit_type == 1:  # Wall
                    color = self.colors["wall"]
                    # Shade based on distance
                    shade = max(0.3, 1.0 - hit_dist / 10)
                    color = [int(c * shade) for c in color]
                elif hit_type == 2:  # Obstacle
                    color = self.colors["obstacle"]

                # Draw vertical strip
                top = (self.img_size - wall_height) // 2
                bottom = top + wall_height
                img[top:bottom, i] = color

        # Check if goal is visible
        goal_vec = self.goal_pos - self.agent_pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.agent_angle
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
        goal_dist = np.linalg.norm(goal_vec)

        if abs(goal_angle) < self.fov / 2:
            # Goal is in field of view
            x = int((goal_angle / self.fov + 0.5) * self.img_size)
            x = np.clip(x, 0, self.img_size - 1)

            # Draw goal marker
            marker_size = max(2, int(10 / (goal_dist + 1)))
            y_center = self.img_size // 2

            img[
                max(0, y_center - marker_size) : min(
                    self.img_size, y_center + marker_size
                ),
                max(0, x - marker_size) : min(self.img_size, x + marker_size),
            ] = self.colors["goal"]

        return img

    def _cast_ray(self, start, angle, max_dist=10, step=0.1):
        """Cast a ray and return distance to first hit and type."""
        for dist in np.arange(0, max_dist, step):
            pos = start + dist * np.array([np.cos(angle), np.sin(angle)])
            gx, gy = int(pos[0]), int(pos[1])

            if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
                return dist, 1  # Hit boundary (wall)

            cell = self.grid[gy, gx]
            if cell > 0:
                return dist, cell

        return -1, 0  # No hit

    def render_topdown(self):
        """Render top-down view for visualization."""
        cell_size = 20
        img = np.full(
            (self.grid_size * cell_size, self.grid_size * cell_size, 3),
            self.colors["floor"],
            dtype=np.uint8,
        )

        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == 1:
                    color = self.colors["wall"]
                elif self.grid[y, x] == 2:
                    color = self.colors["obstacle"]
                else:
                    continue

                x1, y1 = x * cell_size, y * cell_size
                img[y1 : y1 + cell_size, x1 : x1 + cell_size] = color

        # Draw goal
        gx, gy = int(self.goal_pos[0] * cell_size), int(self.goal_pos[1] * cell_size)
        cv2.circle(img, (gx, gy), cell_size // 2, self.colors["goal"], -1)

        # Draw agent
        ax, ay = int(self.agent_pos[0] * cell_size), int(self.agent_pos[1] * cell_size)
        cv2.circle(img, (ax, ay), cell_size // 3, self.colors["agent"], -1)

        # Draw heading
        dx = int(cell_size * np.cos(self.agent_angle))
        dy = int(cell_size * np.sin(self.agent_angle))
        cv2.line(img, (ax, ay), (ax + dx, ay + dy), self.colors["agent"], 2)

        return img


# ============================================================================
# Neural Navigation Policy
# ============================================================================


class NavigationPolicy(nn.Module):
    """
    CNN + LSTM policy for visual navigation.

    - CNN: Process egocentric visual input
    - Goal encoder: Process relative goal position
    - LSTM: Maintain spatial memory
    - Policy head: Output action distribution
    - Value head: For actor-critic training
    """

    def __init__(self, img_size=64, action_dim=3, hidden_dim=128):
        super(NavigationPolicy, self).__init__()

        # Visual encoder (CNN)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),  # 64 -> 15
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 15 -> 6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # 6 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
        )

        # Goal encoder
        self.goal_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU())

        # LSTM for memory
        self.lstm = nn.LSTMCell(256 + 32, hidden_dim)

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.hidden_dim = hidden_dim

    def forward(self, image, goal, hx=None, cx=None):
        """
        Forward pass.

        Args:
            image: (batch, 3, H, W) visual observation
            goal: (batch, 2) relative goal direction
            hx, cx: LSTM hidden states (optional)

        Returns:
            action_logits, value, (hx, cx)
        """
        batch_size = image.shape[0]

        # Initialize hidden states if needed
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_dim).to(image.device)
            cx = torch.zeros(batch_size, self.hidden_dim).to(image.device)

        # Encode visual input
        vis_feat = self.visual_encoder(image)

        # Encode goal
        goal_feat = self.goal_encoder(goal)

        # Combine and pass through LSTM
        combined = torch.cat([vis_feat, goal_feat], dim=-1)
        hx, cx = self.lstm(combined, (hx, cx))

        # Get policy and value
        action_logits = self.policy_head(hx)
        value = self.value_head(hx).squeeze(-1)

        return action_logits, value, hx, cx

    def get_action(self, image, goal, hx=None, cx=None, deterministic=False):
        """Get action from policy."""
        action_logits, value, hx, cx = self.forward(image, goal, hx, cx)

        probs = F.softmax(action_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = F.log_softmax(action_logits, dim=-1)
        log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, value, hx, cx


# ============================================================================
# Training with PPO
# ============================================================================


class PPOTrainer:
    """PPO trainer for navigation policy."""

    def __init__(
        self,
        policy,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages).float()
        returns = advantages + torch.tensor(values).float()

        return advantages, returns

    def update(self, trajectories, epochs=4, batch_size=32):
        """Update policy using collected trajectories."""
        # Flatten trajectories
        states = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []

        for traj in trajectories:
            images = torch.stack(traj["images"])
            goals = torch.stack(traj["goals"])
            acts = torch.tensor(traj["actions"])
            log_probs = torch.stack(traj["log_probs"])
            vals = traj["values"]
            rewards = traj["rewards"]
            dones = traj["dones"]

            adv, ret = self.compute_gae(rewards, vals, dones)

            states.append((images, goals))
            actions.append(acts)
            old_log_probs.append(log_probs)
            advantages.append(adv)
            returns.append(ret)

        # Concatenate
        images = torch.cat([s[0] for s in states])
        goals = torch.cat([s[1] for s in states])
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs)
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        n_updates = 0

        for _ in range(epochs):
            perm = torch.randperm(len(images))

            for i in range(0, len(images), batch_size):
                idx = perm[i : i + batch_size]

                batch_images = images[idx].to(device)
                batch_goals = goals[idx].to(device)
                batch_actions = actions[idx].to(device)
                batch_old_log_probs = old_log_probs[idx].to(device)
                batch_advantages = advantages[idx].to(device)
                batch_returns = returns[idx].to(device)

                # Forward pass (without LSTM state for simplicity)
                action_logits, values, _, _ = self.policy(batch_images, batch_goals)

                # New log probs
                log_probs = F.log_softmax(action_logits, dim=-1)
                new_log_probs = log_probs.gather(
                    1, batch_actions.unsqueeze(-1)
                ).squeeze(-1)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages, clip_ratio * batch_advantages
                ).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                probs = F.softmax(action_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(n_updates, 1)


# ============================================================================
# Training Loop
# ============================================================================


def collect_trajectory(env, policy, max_steps=100):
    """Collect one episode trajectory."""
    obs, _ = env.reset()

    trajectory = {
        "images": [],
        "goals": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }

    hx, cx = None, None

    for _ in range(max_steps):
        # Prepare observation
        image = (
            torch.FloatTensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
            / 255.0
        )
        goal = torch.FloatTensor(obs["goal_direction"]).unsqueeze(0).to(device)

        # Get action
        with torch.no_grad():
            action, log_prob, value, hx, cx = policy.get_action(image, goal, hx, cx)

        # Store
        trajectory["images"].append(image.squeeze(0).cpu())
        trajectory["goals"].append(goal.squeeze(0).cpu())
        trajectory["actions"].append(action.item())
        trajectory["log_probs"].append(log_prob.cpu())
        trajectory["values"].append(value.item())

        # Step
        obs, reward, done, truncated, _ = env.step(action.item())

        trajectory["rewards"].append(reward)
        trajectory["dones"].append(float(done or truncated))

        if done or truncated:
            break

    return trajectory


def train_navigation(num_episodes=500, episodes_per_update=10):
    """Train navigation policy."""

    print("=" * 60)
    print("Demo 11: Neural Navigation")
    print("=" * 60)
    print()

    # Create environment and policy
    env = VisualNavigationEnv(grid_size=12, img_size=64, max_steps=100)
    policy = NavigationPolicy(img_size=64, action_dim=3, hidden_dim=128).to(device)
    trainer = PPOTrainer(policy, lr=3e-4)

    print(f"Environment: VisualNavigationEnv (12x12 grid)")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()

    # Training loop
    rewards_history = []
    success_history = []

    print("Training neural navigation policy...")

    episode = 0
    with tqdm(total=num_episodes, desc="Training") as pbar:
        while episode < num_episodes:
            # Collect trajectories
            trajectories = []
            batch_rewards = []
            batch_success = []

            for _ in range(episodes_per_update):
                traj = collect_trajectory(env, policy)
                trajectories.append(traj)

                ep_reward = sum(traj["rewards"])
                ep_success = ep_reward > 5  # Reached goal

                batch_rewards.append(ep_reward)
                batch_success.append(ep_success)

                episode += 1
                if episode >= num_episodes:
                    break

            rewards_history.extend(batch_rewards)
            success_history.extend(batch_success)

            # Update policy
            loss = trainer.update(trajectories)

            # Update progress
            pbar.update(len(batch_rewards))
            pbar.set_postfix(
                {
                    "reward": np.mean(batch_rewards[-10:]),
                    "success": np.mean(batch_success[-10:]) * 100,
                }
            )

    return policy, env, rewards_history, success_history


def visualize_navigation(policy, env, rewards_history, success_history):
    """Visualize navigation results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards_history, alpha=0.3, color="blue")
    window = 20
    if len(rewards_history) >= window:
        moving_avg = np.convolve(
            rewards_history, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(rewards_history)),
            moving_avg,
            color="red",
            linewidth=2,
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.grid(True, alpha=0.3)

    # 2. Success rate
    ax2 = axes[0, 1]
    if len(success_history) >= window:
        success_rate = (
            np.convolve(
                [float(s) for s in success_history],
                np.ones(window) / window,
                mode="valid",
            )
            * 100
        )
        ax2.plot(
            range(window - 1, len(success_history)),
            success_rate,
            color="green",
            linewidth=2,
        )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Navigation Success Rate")
    ax2.grid(True, alpha=0.3)

    # 3-6. Sample episode visualization
    obs, _ = env.reset(seed=999)

    frames_ego = [obs["image"]]
    frames_topdown = [env.render_topdown()]

    hx, cx = None, None

    for _ in range(50):
        image = (
            torch.FloatTensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
            / 255.0
        )
        goal = torch.FloatTensor(obs["goal_direction"]).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, _, hx, cx = policy.get_action(
                image, goal, hx, cx, deterministic=True
            )

        obs, _, done, _, _ = env.step(action.item())

        frames_ego.append(obs["image"])
        frames_topdown.append(env.render_topdown())

        if done:
            break

    # Show egocentric views
    ax3 = axes[0, 2]
    ax3.imshow(frames_ego[0])
    ax3.set_title("Egocentric View (Start)")
    ax3.axis("off")

    # Top-down views
    indices = [0, len(frames_topdown) // 2, -1]
    titles = ["Start", "Middle", "End"]

    for i, (idx, title) in enumerate(zip(indices, titles)):
        ax = axes[1, i]
        ax.imshow(frames_topdown[idx])
        ax.set_title(f"Top-Down View ({title})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("neural_navigation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'neural_navigation_results.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Train
    policy, env, rewards_history, success_history = train_navigation(
        num_episodes=300, episodes_per_update=5
    )

    # Visualize
    visualize_navigation(policy, env, rewards_history, success_history)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(
        """
1. CNN processes egocentric visual observations
2. Goal direction provides relative target information
3. LSTM maintains spatial memory across timesteps
4. Policy learns to navigate without explicit mapping
5. End-to-end learning from pixels to navigation actions

Comparison with classical methods:
- Classical SLAM: Explicit map + A* planning
- Neural navigation: Implicit map in LSTM state

This approach scales to:
- Complex 3D environments (Habitat, AI2-THOR)
- Real robot navigation
- Object-goal and room-goal navigation

Key papers:
- Neural SLAM (2020): Differentiable mapping + planning
- Habitat Challenge: Large-scale navigation benchmarks
- Object-Nav: Navigate to "find the refrigerator"
    """
    )

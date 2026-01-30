"""
Demo 05: Visual DQN - Learning from Pixels
==========================================

This demo shows DQN learning directly from pixel observations,
similar to the original Atari work but on a simpler custom environment.

Key Concepts:
- Convolutional neural network processes visual input
- Frame stacking captures temporal information (motion)
- Same DQN algorithm but with visual encoder
- End-to-end learning from pixels to actions

Environment: VisualGridWorld
- Agent (blue) must reach goal (green) while avoiding obstacles (red)
- RGB image observation (84x84x3, following Atari preprocessing)
- 4 actions: up, down, left, right

This demonstrates the core visual RL paradigm that revolutionized the field.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Visual Grid World Environment
# ============================================================================


class VisualGridWorld:
    """
    A simple grid world environment with RGB pixel observations.

    - Agent (blue square) starts at a random position
    - Goal (green square) is at a fixed position
    - Obstacles (red squares) block movement
    - Episode ends when agent reaches goal or max steps exceeded
    """

    def __init__(self, grid_size=8, img_size=84, num_obstacles=3, max_steps=100):
        """
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            img_size: Size of rendered image (img_size x img_size)
            num_obstacles: Number of obstacles to place
            max_steps: Maximum steps per episode
        """
        self.grid_size = grid_size
        self.img_size = img_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        self.action_space_n = 4  # up, down, left, right
        self.observation_shape = (img_size, img_size, 3)

        # Colors (RGB)
        self.colors = {
            "agent": [0, 0, 255],  # Blue
            "goal": [0, 255, 0],  # Green
            "obstacle": [255, 0, 0],  # Red
            "background": [40, 40, 40],  # Dark gray
            "grid": [60, 60, 60],  # Lighter gray grid lines
        }

        self.reset()

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0

        # Place goal in bottom-right area
        self.goal_pos = (self.grid_size - 2, self.grid_size - 2)

        # Place obstacles randomly (avoiding goal)
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if pos != self.goal_pos:
                self.obstacles.add(pos)

        # Place agent randomly (avoiding goal and obstacles)
        while True:
            self.agent_pos = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            ]
            pos_tuple = tuple(self.agent_pos)
            if pos_tuple != self.goal_pos and pos_tuple not in self.obstacles:
                break

        return self._render(), {}

    def step(self, action):
        """Take an action in the environment."""
        self.steps += 1

        # Compute new position
        new_pos = self.agent_pos.copy()
        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)

        # Check for obstacle collision
        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos

        # Calculate reward and check termination
        done = False
        truncated = False

        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10.0  # Reached goal
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0  # Timeout penalty
            truncated = True
        else:
            # Small penalty for each step to encourage efficiency
            # Plus distance-based shaping
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
                self.agent_pos[1] - self.goal_pos[1]
            )
            reward = -0.1 - 0.01 * dist

        obs = self._render()
        return obs, reward, done, truncated, {}

    def _render(self):
        """Render the environment as an RGB image."""
        # Create base image
        cell_size = self.img_size // self.grid_size
        img = np.full(
            (self.img_size, self.img_size, 3), self.colors["background"], dtype=np.uint8
        )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = i * cell_size
            img[pos : pos + 1, :] = self.colors["grid"]
            img[:, pos : pos + 1] = self.colors["grid"]

        # Draw obstacles
        for ox, oy in self.obstacles:
            x1, y1 = ox * cell_size + 2, oy * cell_size + 2
            x2, y2 = x1 + cell_size - 4, y1 + cell_size - 4
            img[y1:y2, x1:x2] = self.colors["obstacle"]

        # Draw goal
        gx, gy = self.goal_pos
        x1, y1 = gx * cell_size + 2, gy * cell_size + 2
        x2, y2 = x1 + cell_size - 4, y1 + cell_size - 4
        img[y1:y2, x1:x2] = self.colors["goal"]

        # Draw agent
        ax, ay = self.agent_pos
        x1, y1 = ax * cell_size + 2, ay * cell_size + 2
        x2, y2 = x1 + cell_size - 4, y1 + cell_size - 4
        img[y1:y2, x1:x2] = self.colors["agent"]

        return img


# ============================================================================
# Frame Stacking Wrapper
# ============================================================================


class FrameStack:
    """
    Stack consecutive frames to capture motion/temporal information.

    This is crucial for visual RL because:
    - A single frame doesn't show velocity
    - Stacked frames reveal motion direction and speed
    - Original DQN used 4 stacked frames
    """

    def __init__(self, env, num_frames=4):
        self.env = env
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

        # Update observation shape
        h, w, c = env.observation_shape
        self.observation_shape = (num_frames, h, w)  # Channels first for PyTorch

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        # Convert to grayscale and normalize
        obs = self._preprocess(obs)
        # Initialize frame stack with copies of first frame
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._preprocess(obs)
        self.frames.append(obs)
        return self._get_observation(), reward, done, truncated, info

    def _preprocess(self, obs):
        """Convert RGB to grayscale and normalize."""
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Normalize to [0, 1]
        return gray.astype(np.float32) / 255.0

    def _get_observation(self):
        """Stack frames along first dimension."""
        return np.array(self.frames)


# ============================================================================
# Convolutional Q-Network (Visual DQN)
# ============================================================================


class VisualQNetwork(nn.Module):
    """
    Convolutional Q-Network for visual observations.

    Architecture (following DQN paper):
    - Conv layers extract visual features
    - FC layers compute Q-values from features

    Note: Original DQN used 84x84 grayscale images.
    We follow the same architecture pattern.
    """

    def __init__(self, input_channels, action_dim):
        """
        Args:
            input_channels: Number of stacked frames (typically 4)
            action_dim: Number of actions
        """
        super(VisualQNetwork, self).__init__()

        # Convolutional layers (similar to Nature DQN)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size after convolutions
        # Input: 84x84 -> conv1: 20x20 -> conv2: 9x9 -> conv3: 7x7
        conv_out_size = 64 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        """
        Forward pass: stacked frames -> Q-values.

        Args:
            x: (batch, num_frames, height, width) tensor

        Returns:
            q_values: (batch, action_dim) tensor
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values


# ============================================================================
# Visual DQN Agent
# ============================================================================

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class VisualReplayBuffer:
    """Replay buffer for visual observations."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(batch.action).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(batch.done).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class VisualDQNAgent:
    """DQN Agent for visual observations."""

    def __init__(self, input_channels, action_dim, config):
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.998)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 500)
        self.buffer_size = config.get("buffer_size", 50000)

        self.epsilon = self.epsilon_start

        # Networks
        self.q_network = VisualQNetwork(input_channels, action_dim).to(device)
        self.target_network = VisualQNetwork(input_channels, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = VisualReplayBuffer(self.buffer_size)

        self.update_count = 0
        self.loss_history = []

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.loss_history.append(loss.item())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
# Training
# ============================================================================


def train_visual_dqn(num_episodes=1000, grid_size=8):
    """Train Visual DQN on the grid world."""

    print("=" * 60)
    print("Demo 05: Visual DQN - Learning from Pixels")
    print("=" * 60)
    print()

    # Create environment with frame stacking
    base_env = VisualGridWorld(grid_size=grid_size, img_size=84, num_obstacles=5)
    env = FrameStack(base_env, num_frames=4)

    print(f"Environment: VisualGridWorld {grid_size}x{grid_size}")
    print(f"Observation shape: {env.observation_shape}")
    print(f"Action space: 4 (up, down, left, right)")
    print()

    # Configuration
    config = {
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.998,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "target_update_freq": 500,
        "buffer_size": 50000,
    }

    # Create agent
    agent = VisualDQNAgent(
        input_channels=4, action_dim=4, config=config  # 4 stacked frames
    )

    print(
        f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}"
    )
    print()

    # Training loop
    rewards_history = []
    success_history = []

    print("Training Visual DQN...")

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.update()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_history.append(episode_reward)
        success_history.append(1 if done else 0)  # Success if reached goal

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            print(
                f"\nEpisode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                f"Success Rate = {success_rate:.1f}%, Epsilon = {agent.epsilon:.3f}"
            )

    return agent, env, rewards_history, success_history


def visualize_visual_dqn(agent, env, rewards_history, success_history):
    """Visualize training results and agent behavior."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards_history, alpha=0.3, color="blue")
    window = 50
    if len(rewards_history) >= window:
        moving_avg = np.convolve(
            rewards_history, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(rewards_history)),
            moving_avg,
            color="red",
            linewidth=2,
            label=f"{window}-ep moving avg",
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Success rate
    ax2 = axes[0, 1]
    window = 50
    if len(success_history) >= window:
        success_rate = (
            np.convolve(success_history, np.ones(window) / window, mode="valid") * 100
        )
        ax2.plot(
            range(window - 1, len(success_history)),
            success_rate,
            color="green",
            linewidth=2,
        )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Goal Reaching Success Rate")
    ax2.grid(True, alpha=0.3)

    # 3. Loss history
    ax3 = axes[0, 2]
    if len(agent.loss_history) > 0:
        ax3.plot(agent.loss_history, alpha=0.3, color="purple")
        window = 500
        if len(agent.loss_history) >= window:
            smooth = np.convolve(
                agent.loss_history, np.ones(window) / window, mode="valid"
            )
            ax3.plot(
                range(window - 1, len(agent.loss_history)),
                smooth,
                color="darkviolet",
                linewidth=2,
            )
    ax3.set_xlabel("Update Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Loss")
    ax3.grid(True, alpha=0.3)

    # 4-6. Sample episode visualization
    state, _ = env.reset(seed=999)
    frames = [env.env._render()]  # Get RGB frame from base env
    actions_taken = []

    done = False
    truncated = False
    max_vis_steps = 20
    step = 0

    while not (done or truncated) and step < max_vis_steps:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)
        state, _, done, truncated, _ = env.step(action)
        frames.append(env.env._render())
        step += 1

    # Show first, middle, and last frames
    indices = [0, len(frames) // 2, len(frames) - 1]
    titles = ["Start", "Middle", "End"]
    action_names = ["Up", "Down", "Left", "Right"]

    for i, (idx, title) in enumerate(zip(indices, titles)):
        ax = axes[1, i]
        ax.imshow(frames[idx])
        ax.set_title(f"{title} (Step {idx})")
        ax.axis("off")
        if idx > 0 and idx - 1 < len(actions_taken):
            ax.set_xlabel(f"Action: {action_names[actions_taken[idx-1]]}")

    plt.tight_layout()
    plt.savefig("visual_dqn_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'visual_dqn_results.png'")


def visualize_conv_filters(agent):
    """Visualize learned convolutional filters."""

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("First Convolutional Layer Filters (32 filters, 8x8)", fontsize=14)

    # Get first conv layer weights
    weights = agent.q_network.conv1.weight.data.cpu().numpy()
    # weights shape: (32, 4, 8, 8) - 32 filters, 4 input channels, 8x8 kernel

    # Average across input channels for visualization
    weights_avg = weights.mean(axis=1)  # (32, 8, 8)

    for i in range(32):
        row, col = i // 8, i % 8
        ax = axes[row, col]
        ax.imshow(weights_avg[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("conv_filters.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Filters saved to 'conv_filters.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Train
    agent, env, rewards_history, success_history = train_visual_dqn(
        num_episodes=1000, grid_size=8
    )

    # Visualize
    visualize_visual_dqn(agent, env, rewards_history, success_history)

    # Visualize learned filters
    visualize_conv_filters(agent)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. Visual DQN uses CNNs to process raw pixel observations
2. Frame stacking (4 frames) captures motion information
3. Same DQN algorithm works with visual encoder
4. Convolutional layers learn meaningful visual features
5. End-to-end learning from pixels to actions

This is the core paradigm that enabled:
- Atari game playing (DQN, 2015)
- Visual robotic control
- Autonomous driving from cameras

The visual encoder learns to extract task-relevant features
without hand-crafted engineering - a key breakthrough!
    """)

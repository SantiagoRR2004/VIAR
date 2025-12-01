"""
Demo 04: Deep Q-Network (DQN) on CartPole
==========================================

This demo implements DQN, the algorithm that started the deep RL revolution
by achieving human-level performance on Atari games from raw pixels.

For simplicity, we use CartPole which has a low-dimensional state,
then Demo 05 shows visual DQN from pixels.

Key Concepts:
- Q-learning with neural network function approximation
- Experience replay buffer for stable training
- Target network for stable bootstrap targets
- Epsilon-greedy exploration

Historical Context:
- Mnih et al. (2013): "Playing Atari with Deep RL" - arXiv version
- Mnih et al. (2015): Nature paper - human-level control
- Key insight: experience replay + target networks make deep Q-learning stable

Reference: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Experience Replay Buffer
# ============================================================================

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Key insight from DQN paper:
    - Neural networks expect i.i.d. data
    - Sequential RL data is highly correlated
    - Replay buffer breaks temporal correlations

    Also enables:
    - Reusing rare experiences multiple times
    - More efficient use of data
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
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


# ============================================================================
# Q-Network Architecture
# ============================================================================


class QNetwork(nn.Module):
    """
    Q-Network: Neural network that estimates Q(s, a) for all actions.

    Architecture for CartPole (low-dimensional state):
    - Input: state vector (4 dimensions)
    - Hidden: 2 fully connected layers with ReLU
    - Output: Q-values for each action (2 actions)

    For visual inputs (Demo 05), we would add convolutional layers.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass: state -> Q-values for all actions.

        Args:
            state: (batch, state_dim) tensor

        Returns:
            q_values: (batch, action_dim) tensor
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# ============================================================================
# DQN Agent
# ============================================================================


class DQNAgent:
    """
    DQN Agent implementing the complete algorithm from the Nature paper.
    """

    def __init__(self, state_dim, action_dim, config):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            config: Dictionary of hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)
        self.buffer_size = config.get("buffer_size", 10000)

        # Current epsilon
        self.epsilon = self.epsilon_start

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Training stats
        self.update_count = 0
        self.loss_history = []

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        With probability epsilon: random action (exploration)
        With probability 1-epsilon: greedy action (exploitation)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def update(self):
        """
        Perform one gradient update on the Q-network.

        This is the core of DQN:
        1. Sample batch from replay buffer
        2. Compute target Q-values using target network
        3. Compute loss between predicted and target Q-values
        4. Backpropagate and update Q-network
        5. Periodically update target network
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values: Q(s, a) for the actions that were taken
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss: MSE between current and target Q-values
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.loss_history.append(loss.item())
        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
# Training Loop
# ============================================================================


def train_dqn(env_name="CartPole-v1", num_episodes=500, render_freq=100):
    """
    Train DQN on CartPole environment.

    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        render_freq: Render every N episodes (0 to disable)

    Returns:
        agent: Trained DQN agent
        rewards_history: List of episode rewards
    """
    print("=" * 60)
    print("Demo 04: Deep Q-Network (DQN)")
    print("=" * 60)
    print()

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print()

    # DQN configuration
    config = {
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "target_update_freq": 100,
        "buffer_size": 10000,
    }

    print("DQN Hyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Create agent
    agent = DQNAgent(state_dim, action_dim, config)

    # Training
    rewards_history = []
    epsilon_history = []

    print("Training DQN...")
    print("(CartPole is solved when average reward >= 475 over 100 episodes)")
    print()

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            action = agent.select_action(state, training=True)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done or truncated)

            # Update network
            agent.update()

            state = next_state
            episode_reward += reward

        # Decay epsilon
        agent.decay_epsilon()

        # Record stats
        rewards_history.append(episode_reward)
        epsilon_history.append(agent.epsilon)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"\nEpisode {episode + 1}: Avg Reward = {avg_reward:.1f}, "
                f"Epsilon = {agent.epsilon:.3f}"
            )

    env.close()

    # Final evaluation
    print("\n" + "-" * 40)
    print("Training Complete!")
    print(f"Final average reward (last 100): {np.mean(rewards_history[-100:]):.1f}")

    return agent, rewards_history, epsilon_history


def evaluate_agent(agent, env_name="CartPole-v1", num_episodes=10, render=True):
    """Evaluate trained agent."""
    print("\n" + "=" * 40)
    print("Evaluating Trained Agent")
    print("=" * 40)

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Reward = {episode_reward}")

    env.close()
    print(f"\nMean evaluation reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    return rewards


# ============================================================================
# Visualization
# ============================================================================


def visualize_training(rewards_history, epsilon_history, agent):
    """Create comprehensive training visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards_history, alpha=0.3, color="blue")
    # Moving average
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
            label=f"{window}-episode moving avg",
        )
    ax1.axhline(y=475, color="green", linestyle="--", label="Solved threshold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards Over Episodes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(epsilon_history, color="orange", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Exploration Rate (Epsilon) Decay")
    ax2.grid(True, alpha=0.3)

    # 3. Loss history
    ax3 = axes[1, 0]
    if len(agent.loss_history) > 0:
        ax3.plot(agent.loss_history, alpha=0.3, color="purple")
        # Smooth loss
        window = 100
        if len(agent.loss_history) >= window:
            smooth_loss = np.convolve(
                agent.loss_history, np.ones(window) / window, mode="valid"
            )
            ax3.plot(
                range(window - 1, len(agent.loss_history)),
                smooth_loss,
                color="darkviolet",
                linewidth=2,
                label=f"Smoothed (window={window})",
            )
        ax3.set_xlabel("Update Step")
        ax3.set_ylabel("Loss")
        ax3.set_title("TD Loss Over Training")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Q-value visualization for different states
    ax4 = axes[1, 1]

    # Generate some sample states
    sample_states = [
        [0, 0, 0, 0],  # Balanced
        [0.5, 0, 0, 0],  # Offset position
        [0, 0, 0.2, 0],  # Tilted pole
        [0, 0, -0.2, 0],  # Tilted other way
        [0.5, 0, 0.2, 0.5],  # Complex state
    ]
    state_labels = ["Balanced", "Offset", "Tilt Right", "Tilt Left", "Complex"]

    with torch.no_grad():
        states_tensor = torch.FloatTensor(sample_states).to(device)
        q_values = agent.q_network(states_tensor).cpu().numpy()

    x = np.arange(len(sample_states))
    width = 0.35

    ax4.bar(
        x - width / 2,
        q_values[:, 0],
        width,
        label="Action 0 (Left)",
        color="blue",
        alpha=0.7,
    )
    ax4.bar(
        x + width / 2,
        q_values[:, 1],
        width,
        label="Action 1 (Right)",
        color="red",
        alpha=0.7,
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(state_labels, rotation=15)
    ax4.set_ylabel("Q-value")
    ax4.set_title("Q-values for Sample States")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("dqn_training_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'dqn_training_results.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Train DQN
    agent, rewards_history, epsilon_history = train_dqn(
        env_name="CartPole-v1", num_episodes=500, render_freq=0
    )

    # Visualize training
    visualize_training(rewards_history, epsilon_history, agent)

    # Evaluate (without rendering for faster execution)
    # Set render=True to see the agent play
    eval_rewards = evaluate_agent(agent, num_episodes=10, render=False)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(
        """
1. DQN uses a neural network to approximate Q(s, a)
2. Experience replay breaks temporal correlations in training data
3. Target network provides stable bootstrap targets
4. Epsilon-greedy balances exploration and exploitation
5. The combination of these makes deep Q-learning stable

Key equations:
- Q-learning target: y = r + γ * max_a' Q_target(s', a')
- Loss: L = E[(y - Q(s, a))²]
- Target network update: θ⁻ ← θ every C steps

Next: Demo 05 shows DQN from pixels (visual observations)
    """
    )

"""
Demo 07: Double DQN and Dueling DQN - Addressing DQN's Limitations
==================================================================

This demo implements and compares two major improvements to DQN:
1. Double DQN: Addresses Q-value overestimation
2. Dueling DQN: Decomposes Q into value and advantage

Key Concepts:

Double DQN:
- Standard DQN overestimates Q-values due to max operator
- Solution: Use online network to SELECT action, target to EVALUATE
- y = r + γ * Q_target(s', argmax_a Q_online(s', a))

Dueling DQN:
- Decompose Q(s,a) = V(s) + A(s,a)
- V(s): How good is this state?
- A(s,a): How much better is action a than average?
- Better generalization: V is learned even for unvisited actions

Historical Context:
- Double Q-learning: van Hasselt (2010) for tabular case
- Double DQN: van Hasselt et al. (2016) for deep RL
- Dueling DQN: Wang et al. (2016)
- Both incorporated into Rainbow DQN (2017)

Reference:
- van Hasselt et al., "Deep RL with Double Q-learning" (2016)
- Wang et al., "Dueling Network Architectures for Deep RL" (2016)
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

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Replay Buffer
# ============================================================================

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return (
            torch.FloatTensor(np.array(batch.state)).to(device),
            torch.LongTensor(batch.action).to(device),
            torch.FloatTensor(batch.reward).to(device),
            torch.FloatTensor(np.array(batch.next_state)).to(device),
            torch.FloatTensor(batch.done).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Network Architectures
# ============================================================================


class StandardDQN(nn.Module):
    """
    Standard DQN architecture.
    Simply maps state -> Q(s, a) for all actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.

    Decomposes Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Two streams share initial layers:
    - Value stream: Estimates V(s)
    - Advantage stream: Estimates A(s,a)

    The subtraction of mean advantage ensures identifiability
    and improves stability.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extraction
        self.features = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        features = self.features(x)

        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Q = V + (A - mean(A))
        # Subtracting mean ensures A is zero-mean -> V is the true state value
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def get_value_advantage(self, x):
        """Return value and advantage separately for visualization."""
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


# ============================================================================
# DQN Agent with Different Variants
# ============================================================================


class DQNAgent:
    """
    DQN Agent supporting different variants:
    - 'standard': Original DQN
    - 'double': Double DQN (decouples selection and evaluation)
    - 'dueling': Dueling architecture (V + A decomposition)
    - 'double_dueling': Both improvements combined
    """

    def __init__(self, state_dim, action_dim, variant="standard", config=None):
        self.action_dim = action_dim
        self.variant = variant

        # Default config
        config = config or {}
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)

        # Choose architecture
        use_dueling = "dueling" in variant
        NetworkClass = DuelingDQN if use_dueling else StandardDQN

        self.q_network = NetworkClass(state_dim, action_dim).to(device)
        self.target_network = NetworkClass(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(10000)

        self.update_count = 0
        self.q_value_history = []  # Track Q-values for analysis

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values depend on variant
        with torch.no_grad():
            if "double" in self.variant:
                # Double DQN: Use online network to SELECT, target to EVALUATE
                # This decouples action selection from value estimation
                # Reduces overestimation bias
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = (
                    self.target_network(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                # Standard DQN: Use target network for both
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Track Q-values for analysis
        self.q_value_history.append(current_q.mean().item())

        # Loss and optimization
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================================
# Visualization: Overestimation Analysis
# ============================================================================


def analyze_overestimation(env_name="CartPole-v1", num_episodes=300):
    """
    Compare Q-value estimates across DQN variants.
    Standard DQN tends to overestimate; Double DQN reduces this.
    """
    print("=" * 60)
    print("Analyzing Q-value Overestimation")
    print("=" * 60)
    print()

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Train both variants
    variants = ["standard", "double"]
    results = {}

    for variant in variants:
        print(f"\nTraining {variant.upper()} DQN...")

        agent = DQNAgent(state_dim, action_dim, variant=variant)
        rewards_history = []

        for episode in tqdm(range(num_episodes), desc=variant):
            state, _ = env.reset(seed=SEED + episode)
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, float(done))
                agent.update()

                state = next_state
                episode_reward += reward

            agent.decay_epsilon()
            rewards_history.append(episode_reward)

        results[variant] = {
            "rewards": rewards_history,
            "q_values": agent.q_value_history,
            "agent": agent,
        }

    env.close()

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Learning curves
    ax1 = axes[0, 0]
    for variant, data in results.items():
        rewards = data["rewards"]
        window = 30
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(smoothed, label=f"{variant.upper()} DQN", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Learning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Q-value estimates over training
    ax2 = axes[0, 1]
    for variant, data in results.items():
        q_values = data["q_values"]
        window = 100
        if len(q_values) >= window:
            smoothed = np.convolve(q_values, np.ones(window) / window, mode="valid")
            ax2.plot(smoothed, label=f"{variant.upper()} DQN", linewidth=2)
    ax2.set_xlabel("Update step")
    ax2.set_ylabel("Mean Q-value")
    ax2.set_title(
        "Q-value Estimates Over Training\n(Standard DQN tends to overestimate)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Q-value distribution comparison
    ax3 = axes[1, 0]

    # Generate some test states
    test_states = []
    state, _ = env.reset()
    env = gym.make(env_name)
    for _ in range(100):
        state, _ = env.reset()
        test_states.append(state)
    test_states = torch.FloatTensor(test_states).to(device)

    for variant, data in results.items():
        agent = data["agent"]
        with torch.no_grad():
            q_values = agent.q_network(test_states).max(dim=1)[0].cpu().numpy()
        ax3.hist(q_values, bins=30, alpha=0.5, label=f"{variant.upper()} DQN")

    ax3.set_xlabel("Max Q-value")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Max Q-values on Test States")
    ax3.legend()

    # 4. Text summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = """
    Double DQN Insight:
    
    Standard DQN target:
        y = r + γ * max_a Q_target(s', a)
        
    Problem: max operator causes overestimation
    because we use same values to select AND evaluate
    
    Double DQN target:
        a* = argmax_a Q_online(s', a)   [SELECT]
        y = r + γ * Q_target(s', a*)    [EVALUATE]
        
    Decoupling selection from evaluation
    reduces overestimation bias.
    
    Key insight: Even with noise, Double DQN
    provides more accurate value estimates.
    """

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("double_dqn_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved 'double_dqn_analysis.png'")

    return results


def visualize_dueling_architecture():
    """
    Visualize how Dueling DQN decomposes Q into V and A.
    """
    print("\n" + "=" * 60)
    print("Visualizing Dueling Architecture")
    print("=" * 60)
    print()

    # Create a trained dueling network
    state_dim = 4
    action_dim = 2

    # Train briefly
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim, action_dim, variant="dueling")

    print("Training Dueling DQN briefly...")
    for episode in tqdm(range(200)):
        state, _ = env.reset(seed=SEED + episode)
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state

        agent.decay_epsilon()

    env.close()

    # Analyze V and A decomposition
    print("\nAnalyzing Value-Advantage decomposition...")

    # Generate test states with varying characteristics
    test_states = []
    state_labels = []

    # Balanced state
    test_states.append([0, 0, 0, 0])
    state_labels.append("Balanced")

    # Tilted left (bad)
    test_states.append([0, 0, -0.3, -0.5])
    state_labels.append("Tilting Left")

    # Tilted right (bad)
    test_states.append([0, 0, 0.3, 0.5])
    state_labels.append("Tilting Right")

    # Far left position
    test_states.append([-2, 0, 0, 0])
    state_labels.append("Left Edge")

    # Far right position
    test_states.append([2, 0, 0, 0])
    state_labels.append("Right Edge")

    test_states = torch.FloatTensor(test_states).to(device)

    # Get V and A
    with torch.no_grad():
        values, advantages = agent.q_network.get_value_advantage(test_states)
        q_values = agent.q_network(test_states)

    values = values.cpu().numpy().flatten()
    advantages = advantages.cpu().numpy()
    q_values = q_values.cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. State Values V(s)
    ax1 = axes[0]
    colors = ["green" if v > np.median(values) else "red" for v in values]
    bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(state_labels)))
    ax1.set_xticklabels(state_labels, rotation=45, ha="right")
    ax1.set_ylabel("V(s)")
    ax1.set_title('State Value V(s)\n"How good is this state?"')
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Advantages A(s, a)
    ax2 = axes[1]
    x = np.arange(len(state_labels))
    width = 0.35

    ax2.bar(
        x - width / 2,
        advantages[:, 0],
        width,
        label="Left (a=0)",
        color="blue",
        alpha=0.7,
    )
    ax2.bar(
        x + width / 2,
        advantages[:, 1],
        width,
        label="Right (a=1)",
        color="orange",
        alpha=0.7,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels, rotation=45, ha="right")
    ax2.set_ylabel("A(s, a)")
    ax2.set_title('Action Advantage A(s, a)\n"How much better is this action?"')
    ax2.legend()
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Final Q-values Q(s, a) = V(s) + A(s, a)
    ax3 = axes[2]

    ax3.bar(
        x - width / 2,
        q_values[:, 0],
        width,
        label="Q(s, Left)",
        color="blue",
        alpha=0.7,
    )
    ax3.bar(
        x + width / 2,
        q_values[:, 1],
        width,
        label="Q(s, Right)",
        color="orange",
        alpha=0.7,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(state_labels, rotation=45, ha="right")
    ax3.set_ylabel("Q(s, a)")
    ax3.set_title(
        "Q-values: Q(s,a) = V(s) + A(s,a)\n(Combines state value and action advantage)"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("dueling_architecture.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved 'dueling_architecture.png'")

    # Print decomposition
    print("\nValue-Advantage Decomposition:")
    print("-" * 60)
    for i, label in enumerate(state_labels):
        print(f"{label}:")
        print(f"  V(s) = {values[i]:.3f}")
        print(
            f"  A(s, Left) = {advantages[i, 0]:.3f}, A(s, Right) = {advantages[i, 1]:.3f}"
        )
        print(
            f"  Q(s, Left) = {q_values[i, 0]:.3f}, Q(s, Right) = {q_values[i, 1]:.3f}"
        )
        print()


def compare_all_variants():
    """Compare all DQN variants on CartPole."""
    print("\n" + "=" * 60)
    print("Comparing All DQN Variants")
    print("=" * 60)
    print()

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    variants = ["standard", "double", "dueling", "double_dueling"]
    results = {}
    num_episodes = 400

    for variant in variants:
        print(f"\nTraining {variant.replace('_', ' + ').upper()} DQN...")

        agent = DQNAgent(state_dim, action_dim, variant=variant)
        rewards_history = []

        for episode in tqdm(range(num_episodes), desc=variant):
            state, _ = env.reset(seed=SEED + episode)
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.replay_buffer.push(state, action, reward, next_state, float(done))
                agent.update()
                state = next_state
                episode_reward += reward

            agent.decay_epsilon()
            rewards_history.append(episode_reward)

        results[variant] = rewards_history

    env.close()

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "standard": "red",
        "double": "blue",
        "dueling": "green",
        "double_dueling": "purple",
    }

    window = 30
    for variant, rewards in results.items():
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            label = variant.replace("_", " + ").upper()
            ax.plot(
                smoothed, label=label, color=colors[variant], linewidth=2, alpha=0.8
            )

    ax.axhline(y=475, color="black", linestyle="--", label="Solved threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("DQN Variants Comparison on CartPole-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dqn_variants_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved 'dqn_variants_comparison.png'")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # 1. Analyze overestimation
    analyze_overestimation(num_episodes=300)

    # 2. Visualize dueling architecture
    visualize_dueling_architecture()

    # 3. Compare all variants
    compare_all_variants()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(
        """
Double DQN:
- Standard DQN overestimates Q-values (max is biased estimator)
- Double DQN decouples selection (online) from evaluation (target)
- Target: y = r + γ * Q_target(s', argmax_a Q_online(s', a))
- Reduces overestimation, improves stability

Dueling DQN:
- Decompose Q(s,a) = V(s) + A(s,a) - mean(A)
- V(s): State value (how good is being here?)
- A(s,a): Advantage (how much better is this action than average?)
- Better generalization: V learned even for unvisited actions
- Useful when actions don't always matter (many states, action irrelevant)

Combining both (Double Dueling DQN):
- Get benefits of both improvements
- More accurate values + better generalization
- Foundation for Rainbow DQN (adds 4 more improvements)

Rainbow DQN (2017) combines:
1. Double DQN
2. Dueling architecture
3. Prioritized experience replay
4. Multi-step returns
5. Distributional RL
6. Noisy networks
    """
    )

"""
Demo 06: Experience Replay - Breaking Temporal Correlations
============================================================

This demo visualizes and explains the experience replay buffer,
one of the two key innovations that made DQN stable.

Key Concepts:
- Why naive online learning fails with neural networks
- How replay buffer breaks temporal correlations
- Uniform vs. prioritized sampling
- Buffer dynamics and sample diversity

The Problem:
- Neural networks expect i.i.d. (independent, identically distributed) data
- Sequential RL data is highly correlated (adjacent states are similar)
- Training on correlated data leads to catastrophic forgetting
- Solution: Store transitions and sample randomly

Historical Context:
- Experience replay was introduced by Lin (1992)
- DQN (2015) showed it was crucial for stable deep RL
- Prioritized Experience Replay (2016) improved efficiency
- Now standard in all off-policy deep RL algorithms

Reference: Mnih et al., "Human-level control through deep RL" (2015)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from collections import deque, namedtuple
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ============================================================================
# Replay Buffer Implementations
# ============================================================================

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class UniformReplayBuffer:
    """
    Standard uniform replay buffer as used in DQN.

    - Fixed capacity (circular buffer)
    - Uniform random sampling
    - O(1) insert, O(1) sample
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # Statistics
        self.insert_count = 0
        self.sample_count = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        transition = Transition(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        self.insert_count += 1

    def sample(self, batch_size):
        """Sample a random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        self.sample_count += batch_size
        return batch, indices

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).

    - Samples transitions with probability proportional to TD error
    - High-error transitions are sampled more often
    - Uses importance sampling to correct for bias
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, priority=None):
        """Store a transition with priority."""
        transition = Transition(state, action, reward, next_state, done)

        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample based on priorities."""
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Visualization: Why Replay Matters
# ============================================================================


def visualize_temporal_correlation():
    """Show why sequential data is problematic."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Sequential states are correlated
    ax1 = axes[0]

    # Simulate trajectory through state space
    t = np.linspace(0, 4 * np.pi, 100)
    states_x = np.sin(t) + 0.1 * np.random.randn(100)
    states_y = np.cos(t) + 0.1 * np.random.randn(100)

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))

    for i in range(len(t) - 1):
        ax1.plot(states_x[i : i + 2], states_y[i : i + 2], c=colors[i], linewidth=2)

    ax1.scatter(states_x, states_y, c=range(len(t)), cmap="viridis", s=30, zorder=5)
    ax1.set_xlabel("State dimension 1")
    ax1.set_ylabel("State dimension 2")
    ax1.set_title("Sequential States (Correlated)")
    ax1.set_aspect("equal")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, len(t)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="Time step")

    # 2. Online training: sequential batches
    ax2 = axes[1]

    batch_size = 8
    n_batches = 5

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        y_offset = batch_idx * 2

        for i in range(start, min(end, len(t))):
            ax2.scatter(i, y_offset, c=[colors[i]], s=100)

        ax2.axhline(y=y_offset, color="gray", linestyle="--", alpha=0.3)
        ax2.text(-5, y_offset, f"Batch {batch_idx + 1}", va="center", fontsize=10)

    ax2.set_xlabel("Time step in episode")
    ax2.set_ylabel("Training batch")
    ax2.set_title(
        "Online Learning: Sequential Batches\n(HIGH correlation within batch)"
    )
    ax2.set_xlim(-10, 50)

    # 3. Replay: random batches
    ax3 = axes[2]

    buffer_size = 100

    for batch_idx in range(n_batches):
        # Random sampling from buffer
        indices = np.random.choice(buffer_size, batch_size, replace=False)
        indices.sort()

        y_offset = batch_idx * 2

        for i, idx in enumerate(indices):
            ax3.scatter(idx, y_offset, c=[colors[idx % len(colors)]], s=100)

        ax3.axhline(y=y_offset, color="gray", linestyle="--", alpha=0.3)
        ax3.text(-5, y_offset, f"Batch {batch_idx + 1}", va="center", fontsize=10)

    ax3.set_xlabel("Original time step (shuffled)")
    ax3.set_ylabel("Training batch")
    ax3.set_title("Replay Buffer: Random Batches\n(LOW correlation within batch)")
    ax3.set_xlim(-10, 110)

    plt.tight_layout()
    plt.savefig("replay_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'replay_correlation.png'")


def visualize_buffer_dynamics():
    """Visualize how the replay buffer fills and overwrites."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create buffer and simulate filling
    capacity = 50
    buffer = UniformReplayBuffer(capacity)

    # Generate synthetic episode data
    n_steps = 150
    states = np.cumsum(np.random.randn(n_steps, 2) * 0.3, axis=0)
    actions = np.random.randint(0, 4, n_steps)
    rewards = np.random.randn(n_steps) * 0.5

    # Track buffer contents at different stages
    snapshots = []
    snapshot_times = [25, 50, 100, 150]

    for t in range(n_steps):
        buffer.push(
            states[t], actions[t], rewards[t], states[min(t + 1, n_steps - 1)], False
        )

        if t + 1 in snapshot_times:
            # Store current buffer state (just positions for visualization)
            positions = [i for i in range(len(buffer.buffer))]
            original_times = [
                (buffer.insert_count - len(buffer.buffer) + i) % n_steps
                for i in range(len(buffer.buffer))
            ]
            snapshots.append((t + 1, positions, original_times))

    # Plot snapshots
    titles = [
        f"After {t} insertions (buffer size: {min(t, capacity)})"
        for t in snapshot_times
    ]

    for ax, (t, positions, times), title in zip(axes.flat, snapshots, titles):
        colors = plt.cm.plasma(np.array(times) / n_steps)

        # Draw buffer slots
        for i in range(capacity):
            rect = Rectangle(
                (i, 0),
                0.9,
                1,
                facecolor=(
                    "lightgray" if i >= len(positions) else colors[i % len(colors)]
                ),
                edgecolor="black",
            )
            ax.add_patch(rect)

        # Mark write position
        write_pos = t % capacity
        ax.axvline(x=write_pos, color="red", linewidth=3, label="Write position")

        ax.set_xlim(-1, capacity + 1)
        ax.set_ylim(-0.5, 2)
        ax.set_xlabel("Buffer position")
        ax.set_title(title)
        ax.set_yticks([])

        if ax == axes.flat[0]:
            ax.legend(loc="upper right")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, n_steps))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Original insertion time", shrink=0.6)

    plt.suptitle("Circular Replay Buffer Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig("buffer_dynamics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'buffer_dynamics.png'")


def visualize_prioritized_replay():
    """Compare uniform vs. prioritized replay."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create buffers
    capacity = 100
    uniform_buffer = UniformReplayBuffer(capacity)
    priority_buffer = PrioritizedReplayBuffer(capacity, alpha=0.6)

    # Fill with transitions of varying importance (simulated TD errors)
    np.random.seed(42)

    # Most transitions are "boring" (low TD error)
    # A few are "important" (high TD error)
    td_errors = np.concatenate(
        [
            np.random.exponential(0.1, 90),  # Mostly small errors
            np.random.exponential(2.0, 10),  # A few large errors
        ]
    )
    np.random.shuffle(td_errors)

    for i in range(capacity):
        state = np.array([i, td_errors[i]])
        uniform_buffer.push(state, 0, td_errors[i], state, False)
        priority_buffer.push(
            state, 0, td_errors[i], state, False, priority=td_errors[i]
        )

    # 1. TD error distribution
    ax1 = axes[0]
    ax1.bar(range(capacity), td_errors, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Transition index")
    ax1.set_ylabel("TD Error (importance)")
    ax1.set_title("TD Error Distribution in Buffer")
    ax1.axhline(
        y=np.mean(td_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(td_errors):.2f}",
    )
    ax1.legend()

    # 2. Sampling distribution - Uniform
    ax2 = axes[1]

    n_samples = 10000
    sample_counts_uniform = np.zeros(capacity)

    for _ in range(n_samples // 32):
        _, indices = uniform_buffer.sample(32)
        for idx in indices:
            sample_counts_uniform[idx] += 1

    ax2.bar(range(capacity), sample_counts_uniform, color="green", alpha=0.7)
    ax2.set_xlabel("Transition index")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Uniform Sampling Distribution\n(All transitions sampled equally)")

    # 3. Sampling distribution - Prioritized
    ax3 = axes[2]

    sample_counts_priority = np.zeros(capacity)

    for _ in range(n_samples // 32):
        _, indices, _ = priority_buffer.sample(32)
        for idx in indices:
            sample_counts_priority[idx] += 1

    # Color by TD error
    colors = plt.cm.Reds(td_errors / td_errors.max())
    ax3.bar(range(capacity), sample_counts_priority, color=colors, alpha=0.7)
    ax3.set_xlabel("Transition index")
    ax3.set_ylabel("Sample count")
    ax3.set_title("Prioritized Sampling Distribution\n(High TD-error sampled more)")

    plt.tight_layout()
    plt.savefig("prioritized_replay.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'prioritized_replay.png'")


def visualize_sample_diversity():
    """Show how replay increases sample diversity over time."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate training with and without replay
    n_episodes = 20
    steps_per_episode = 50
    batch_size = 32

    # Track which states are seen in each training batch
    # Without replay: sequential states
    online_diversity = []

    for ep in range(n_episodes):
        for t in range(0, steps_per_episode, batch_size):
            # States in this batch all from same time window
            batch_states = list(
                range(
                    ep * steps_per_episode + t, ep * steps_per_episode + t + batch_size
                )
            )
            # Diversity = number of unique episodes represented
            episodes_in_batch = set([s // steps_per_episode for s in batch_states])
            online_diversity.append(len(episodes_in_batch))

    # With replay: random states from buffer
    buffer_size = 1000
    replay_diversity = []
    buffer = list(range(min(100, buffer_size)))  # Start with some data

    for ep in range(n_episodes):
        for t in range(0, steps_per_episode, batch_size):
            # Add new state to buffer
            new_state = ep * steps_per_episode + t
            if len(buffer) < buffer_size:
                buffer.append(new_state)
            else:
                buffer[np.random.randint(buffer_size)] = new_state

            # Sample random batch
            if len(buffer) >= batch_size:
                batch_indices = np.random.choice(len(buffer), batch_size, replace=False)
                batch_states = [buffer[i] for i in batch_indices]
                episodes_in_batch = set([s // steps_per_episode for s in batch_states])
                replay_diversity.append(len(episodes_in_batch))

    # Plot
    ax1 = axes[0]
    ax1.plot(online_diversity, "r-", alpha=0.7, label="Online (no replay)")
    ax1.plot(replay_diversity, "b-", alpha=0.7, label="With replay buffer")
    ax1.set_xlabel("Training batch")
    ax1.set_ylabel("Episodes represented in batch")
    ax1.set_title("Sample Diversity: Episodes per Batch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Moving average
    ax2 = axes[1]
    window = 20

    if len(online_diversity) >= window:
        online_ma = np.convolve(
            online_diversity, np.ones(window) / window, mode="valid"
        )
        replay_ma = np.convolve(
            replay_diversity, np.ones(window) / window, mode="valid"
        )

        ax2.plot(online_ma, "r-", linewidth=2, label="Online (no replay)")
        ax2.plot(replay_ma, "b-", linewidth=2, label="With replay buffer")

    ax2.set_xlabel("Training batch")
    ax2.set_ylabel("Episodes represented (smoothed)")
    ax2.set_title("Sample Diversity Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sample_diversity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'sample_diversity.png'")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("Demo 06: Experience Replay - Breaking Temporal Correlations")
    print("=" * 60)
    print()

    print("1. Visualizing why temporal correlation is problematic...")
    visualize_temporal_correlation()
    print()

    print("2. Visualizing replay buffer dynamics...")
    visualize_buffer_dynamics()
    print()

    print("3. Comparing uniform vs. prioritized replay...")
    visualize_prioritized_replay()
    print()

    print("4. Showing sample diversity improvement...")
    visualize_sample_diversity()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. Sequential RL data is highly correlated (bad for neural networks)
2. Replay buffer stores transitions and samples randomly
3. Random sampling breaks temporal correlations -> i.i.d. batches
4. Enables reusing rare/important experiences multiple times
5. Prioritized replay focuses on high TD-error transitions

Without replay:
- Training is unstable
- Network forgets early experiences
- Poor sample efficiency

With replay:
- Stable training
- Better sample reuse
- Foundation for all off-policy deep RL

Key equations:
- Uniform: P(transition i) = 1/N
- Prioritized: P(i) ∝ |TD_error_i|^α
- Importance weights: w_i = (N * P(i))^(-β)
    """)


if __name__ == "__main__":
    print(__doc__)
    main()

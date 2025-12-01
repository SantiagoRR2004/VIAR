"""
Demo 10: Simplified Dreamer - Learning to Imagine and Plan
==========================================================

This demo implements a simplified version of Dreamer, which learns
a world model and trains a policy entirely through imagination.

Key Concepts:
- RSSM: Recurrent State-Space Model (deterministic + stochastic states)
- Actor-Critic in latent space (no pixel rendering during training)
- Imagination: Roll out world model to generate training data
- λ-returns: Temporal difference learning with multi-step returns

Dreamer Architecture:
1. World Model:
   - Encoder: image -> posterior latent state
   - Dynamics: predict next latent state from action
   - Decoder: latent state -> predicted image
   - Reward predictor: latent state -> predicted reward

2. Actor-Critic:
   - Actor: latent state -> action distribution
   - Critic: latent state -> value estimate
   - Trained on IMAGINED trajectories (no real environment interaction)

Historical Context:
- Dreamer (Hafner et al., 2020): First to match model-free with world models
- DreamerV2 (2021): Discrete latents, better on Atari
- DreamerV3 (2023): Single hyperparameters across domains

Reference: Hafner et al., "Dream to Control" (ICLR 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Simple Environment
# ============================================================================


class SimplePendulum:
    """
    Simple pendulum environment with visual observations.

    State: angle θ, angular velocity θ̇
    Action: torque (-1 to 1)
    Goal: Swing up and balance at θ = 0

    Visual observation: 64x64 RGB image showing pendulum
    """

    def __init__(self, img_size=64):
        self.img_size = img_size
        self.dt = 0.05
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

        self.max_steps = 200
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Start hanging down with small perturbation
        self.theta = np.pi + np.random.uniform(-0.1, 0.1)
        self.theta_dot = np.random.uniform(-0.1, 0.1)
        self.steps = 0

        return self._render()

    def step(self, action):
        self.steps += 1

        # Clip action
        u = np.clip(action, -1, 1) * self.max_torque

        # Physics update (simple Euler)
        theta_ddot = self.g / self.l * np.sin(self.theta) + u / (self.m * self.l**2)
        self.theta_dot = np.clip(
            self.theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed
        )
        self.theta = self.theta + self.theta_dot * self.dt

        # Normalize angle to [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

        # Reward: higher when upright (θ ≈ 0) and slow
        # Angle cost + velocity cost + control cost
        reward = -(self.theta**2 + 0.1 * self.theta_dot**2 + 0.001 * u**2)

        done = self.steps >= self.max_steps

        return self._render(), reward, done, {}

    def _render(self):
        """Render pendulum as image."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Background
        img[:] = [50, 50, 70]

        # Pivot point
        cx, cy = self.img_size // 2, self.img_size // 3

        # Pendulum endpoint
        length_pixels = self.img_size // 3
        ex = int(cx + length_pixels * np.sin(self.theta))
        ey = int(cy + length_pixels * np.cos(self.theta))

        # Draw rod
        cv2.line(img, (cx, cy), (ex, ey), (200, 200, 200), 3)

        # Draw pivot
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

        # Draw bob (color indicates velocity)
        speed_color = min(255, int(abs(self.theta_dot) / self.max_speed * 255))
        bob_color = (255 - speed_color, speed_color, 100)
        cv2.circle(img, (ex, ey), 10, bob_color, -1)

        return img


# ============================================================================
# RSSM: Recurrent State-Space Model
# ============================================================================


class RSSM(nn.Module):
    """
    Recurrent State-Space Model.

    State consists of:
    - Deterministic state h: Captures long-term memory (LSTM-like)
    - Stochastic state s: Captures uncertainty and noise

    Key components:
    - Prior: p(s_t | h_t) - predict stochastic state from deterministic
    - Posterior: q(s_t | h_t, o_t) - infer stochastic state from observation
    - Dynamics: h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) - deterministic transition
    """

    def __init__(
        self, obs_dim, action_dim, hidden_dim=200, stoch_dim=30, deter_dim=200
    ):
        super().__init__()

        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # Dynamics: GRU cell for deterministic state
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # Prior: p(s_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean and std
        )

        # Posterior: q(s_t | h_t, o_t)
        # o_t is encoded observation
        self.posterior_fc = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean and std
        )

    def initial_state(self, batch_size):
        """Initialize hidden states."""
        h = torch.zeros(batch_size, self.deter_dim).to(device)
        s = torch.zeros(batch_size, self.stoch_dim).to(device)
        return h, s

    def get_prior(self, h):
        """Get prior distribution p(s|h)."""
        stats = self.prior_fc(h)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return Normal(mean, std)

    def get_posterior(self, h, obs_embed):
        """Get posterior distribution q(s|h,o)."""
        x = torch.cat([h, obs_embed], dim=-1)
        stats = self.posterior_fc(x)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return Normal(mean, std)

    def forward(self, prev_h, prev_s, action):
        """
        One step of dynamics (imagination).

        Returns new deterministic state and prior over stochastic state.
        """
        x = torch.cat([prev_s, action], dim=-1)
        h = self.gru(x, prev_h)
        prior = self.get_prior(h)
        s = prior.rsample()
        return h, s, prior

    def observe(self, prev_h, prev_s, action, obs_embed):
        """
        One step with observation (learning from real data).

        Returns new states and both prior and posterior distributions.
        """
        x = torch.cat([prev_s, action], dim=-1)
        h = self.gru(x, prev_h)
        prior = self.get_prior(h)
        posterior = self.get_posterior(h, obs_embed)
        s = posterior.rsample()
        return h, s, prior, posterior


# ============================================================================
# World Model Components
# ============================================================================


class Encoder(nn.Module):
    """Encode image observation to embedding."""

    def __init__(self, obs_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 31x31
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 6x6
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 2x2
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256 * 2 * 2, obs_dim)

    def forward(self, obs):
        # obs: (batch, 3, 64, 64)
        h = self.conv(obs)
        return self.fc(h)


class Decoder(nn.Module):
    """Decode latent state to predicted image."""

    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 256 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=1),  # 31x31
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, output_padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, state):
        h = self.fc(state)
        h = h.view(-1, 256, 2, 2)
        return self.deconv(h)


class RewardPredictor(nn.Module):
    """Predict reward from latent state."""

    def __init__(self, state_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


# ============================================================================
# Actor-Critic
# ============================================================================


class Actor(nn.Module):
    """Policy network: state -> action distribution."""

    def __init__(self, state_dim, action_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        std = F.softplus(self.std(h)) + 0.1
        return Normal(mean, std)


class Critic(nn.Module):
    """Value network: state -> value."""

    def __init__(self, state_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ============================================================================
# Dreamer Agent
# ============================================================================


class SimpleDreamer:
    """
    Simplified Dreamer agent.

    Training loop:
    1. Collect real experience with current policy
    2. Train world model on real experience
    3. Imagine trajectories using world model
    4. Train actor-critic on imagined trajectories
    """

    def __init__(
        self,
        action_dim,
        obs_dim=256,
        hidden_dim=200,
        stoch_dim=30,
        deter_dim=200,
        horizon=15,
    ):

        self.action_dim = action_dim
        self.horizon = horizon
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # World model
        self.encoder = Encoder(obs_dim).to(device)
        self.rssm = RSSM(obs_dim, action_dim, hidden_dim, stoch_dim, deter_dim).to(
            device
        )
        self.decoder = Decoder(deter_dim + stoch_dim).to(device)
        self.reward_pred = RewardPredictor(deter_dim + stoch_dim, hidden_dim).to(device)

        # Actor-Critic
        self.actor = Actor(deter_dim + stoch_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(deter_dim + stoch_dim, hidden_dim).to(device)

        # Optimizers
        world_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_pred.parameters())
        )
        self.world_optimizer = optim.Adam(world_params, lr=3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        # Experience buffer
        self.buffer = []

        # Training stats
        self.world_losses = []
        self.actor_losses = []
        self.critic_losses = []

    def get_state(self, h, s):
        """Concatenate deterministic and stochastic state."""
        return torch.cat([h, s], dim=-1)

    def select_action(self, obs, h, s, training=True):
        """Select action given observation and hidden state."""
        obs_tensor = (
            torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )

        with torch.no_grad():
            obs_embed = self.encoder(obs_tensor)
            _, s_new, _, posterior = self.rssm.observe(
                h, s, torch.zeros(1, self.action_dim).to(device), obs_embed
            )

            # Use posterior mean for deterministic state update
            h_new = self.rssm.gru(
                torch.cat([s, torch.zeros(1, self.action_dim).to(device)], dim=-1), h
            )

            state = self.get_state(h_new, s_new)
            action_dist = self.actor(state)

            if training:
                action = action_dist.sample()
            else:
                action = action_dist.mean

            action = torch.tanh(action)  # Bound to [-1, 1]

        return action.cpu().numpy().flatten(), h_new, s_new

    def store_experience(self, episode):
        """Store episode in buffer."""
        self.buffer.append(episode)
        # Keep last 100 episodes
        if len(self.buffer) > 100:
            self.buffer.pop(0)

    def train_world_model(self, batch_size=16, seq_len=20):
        """Train world model on real experience."""
        if len(self.buffer) < 5:
            return

        # Sample sequences
        batch_obs = []
        batch_actions = []
        batch_rewards = []

        for _ in range(batch_size):
            ep = self.buffer[np.random.randint(len(self.buffer))]
            if len(ep["obs"]) <= seq_len:
                continue

            start = np.random.randint(0, len(ep["obs"]) - seq_len)
            batch_obs.append(ep["obs"][start : start + seq_len])
            batch_actions.append(ep["actions"][start : start + seq_len])
            batch_rewards.append(ep["rewards"][start : start + seq_len])

        if len(batch_obs) < 2:
            return

        # Convert to tensors
        obs = (
            torch.FloatTensor(np.array(batch_obs)).permute(0, 1, 4, 2, 3).to(device)
            / 255.0
        )
        actions = torch.FloatTensor(np.array(batch_actions)).to(device)
        rewards = torch.FloatTensor(np.array(batch_rewards)).to(device)

        batch_size, seq_len = obs.shape[:2]

        # Initialize states
        h, s = self.rssm.initial_state(batch_size)

        # Forward through sequence
        recon_loss = 0
        reward_loss = 0
        kl_loss = 0

        for t in range(seq_len):
            obs_embed = self.encoder(obs[:, t])

            if t == 0:
                action = torch.zeros(batch_size, self.action_dim).to(device)
            else:
                action = actions[:, t - 1]

            h, s, prior, posterior = self.rssm.observe(h, s, action, obs_embed)
            state = self.get_state(h, s)

            # Reconstruction loss
            recon = self.decoder(state)
            recon_loss += F.mse_loss(recon, obs[:, t])

            # Reward loss
            reward_pred = self.reward_pred(state)
            reward_loss += F.mse_loss(reward_pred.squeeze(), rewards[:, t])

            # KL loss
            kl_loss += kl_divergence(posterior, prior).sum(dim=-1).mean()

        loss = recon_loss + reward_loss + 0.1 * kl_loss

        self.world_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 100)
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100)
        self.world_optimizer.step()

        self.world_losses.append(loss.item())

    def train_actor_critic(self, batch_size=16):
        """Train actor-critic on imagined trajectories."""
        if len(self.buffer) < 5:
            return

        # Get initial states from real data
        batch_obs = []
        for _ in range(batch_size):
            ep = self.buffer[np.random.randint(len(self.buffer))]
            idx = np.random.randint(len(ep["obs"]))
            batch_obs.append(ep["obs"][idx])

        obs = (
            torch.FloatTensor(np.array(batch_obs)).permute(0, 3, 1, 2).to(device)
            / 255.0
        )

        # Encode initial observations
        with torch.no_grad():
            obs_embed = self.encoder(obs)
            h, s = self.rssm.initial_state(batch_size)
            _, s, _, _ = self.rssm.observe(
                h, s, torch.zeros(batch_size, self.action_dim).to(device), obs_embed
            )
            h = self.rssm.gru(
                torch.cat(
                    [s, torch.zeros(batch_size, self.action_dim).to(device)], dim=-1
                ),
                h,
            )

        # Imagine trajectories
        states = []
        actions = []
        rewards = []

        for t in range(self.horizon):
            state = self.get_state(h, s)
            states.append(state)

            # Sample action from actor
            action_dist = self.actor(state)
            action = torch.tanh(action_dist.rsample())
            actions.append(action)

            # Predict reward
            reward = self.reward_pred(state)
            rewards.append(reward.squeeze())

            # Step dynamics
            h, s, _ = self.rssm(h, s, action)

        # Stack
        states = torch.stack(states, dim=1)  # (batch, horizon, state_dim)
        rewards = torch.stack(rewards, dim=1)  # (batch, horizon)

        # Compute values
        with torch.no_grad():
            values = self.critic(states)

        # Compute λ-returns
        gamma = 0.99
        lambda_ = 0.95

        returns = torch.zeros_like(rewards)
        last_return = values[:, -1]

        for t in reversed(range(self.horizon)):
            returns[:, t] = rewards[:, t] + gamma * (
                (1 - lambda_) * values[:, t] + lambda_ * last_return
            )
            last_return = returns[:, t]

        # Update critic
        values = self.critic(states.detach())
        critic_loss = F.mse_loss(values, returns.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.critic_optimizer.step()

        # Update actor
        values = self.critic(states)
        actor_loss = -values.mean()  # Maximize value

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())


# ============================================================================
# Training and Visualization
# ============================================================================


def train_dreamer(num_episodes=200):
    """Train Dreamer on pendulum environment."""

    print("=" * 60)
    print("Demo 10: Simplified Dreamer")
    print("=" * 60)
    print()

    env = SimplePendulum()
    agent = SimpleDreamer(action_dim=1)

    print("Training Dreamer agent...")
    print("(This may take a few minutes)")
    print()

    rewards_history = []

    for episode in tqdm(range(num_episodes)):
        obs = env.reset(seed=episode)
        h, s = agent.rssm.initial_state(1)

        episode_data = {"obs": [obs], "actions": [], "rewards": []}
        episode_reward = 0

        done = False
        while not done:
            action, h, s = agent.select_action(obs, h, s)
            obs, reward, done, _ = env.step(action[0])

            episode_data["obs"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_reward += reward

        agent.store_experience(episode_data)
        rewards_history.append(episode_reward)

        # Train
        for _ in range(5):
            agent.train_world_model()
            agent.train_actor_critic()

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.1f}")

    return agent, env, rewards_history


def visualize_dreamer(agent, env, rewards_history):
    """Visualize Dreamer results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards_history, alpha=0.3, color="blue")
    window = 20
    if len(rewards_history) >= window:
        smoothed = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(rewards_history)), smoothed, "r-", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.grid(True, alpha=0.3)

    # 2. World model loss
    ax2 = axes[0, 1]
    if len(agent.world_losses) > 0:
        ax2.plot(agent.world_losses, alpha=0.5)
        window = 100
        if len(agent.world_losses) >= window:
            smoothed = np.convolve(
                agent.world_losses, np.ones(window) / window, mode="valid"
            )
            ax2.plot(
                range(window - 1, len(agent.world_losses)), smoothed, "r-", linewidth=2
            )
    ax2.set_xlabel("Update")
    ax2.set_ylabel("Loss")
    ax2.set_title("World Model Loss")
    ax2.grid(True, alpha=0.3)

    # 3. Actor loss
    ax3 = axes[0, 2]
    if len(agent.actor_losses) > 0:
        ax3.plot(agent.actor_losses, alpha=0.5, color="green")
    ax3.set_xlabel("Update")
    ax3.set_ylabel("Loss")
    ax3.set_title("Actor Loss (negative value)")
    ax3.grid(True, alpha=0.3)

    # 4-6. Sample episode frames
    obs = env.reset(seed=999)
    h, s = agent.rssm.initial_state(1)
    frames = [obs]

    for _ in range(50):
        action, h, s = agent.select_action(obs, h, s, training=False)
        obs, _, done, _ = env.step(action[0])
        frames.append(obs)
        if done:
            break

    indices = [0, len(frames) // 2, len(frames) - 1]
    titles = ["Start", "Middle", "End"]

    for i, (idx, title) in enumerate(zip(indices, titles)):
        axes[1, i].imshow(frames[idx])
        axes[1, i].set_title(f"{title} (Step {idx})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("dreamer_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved 'dreamer_results.png'")


def visualize_imagination(agent, env):
    """Visualize imagined vs real trajectories."""

    print("\nVisualizing imagination vs reality...")

    # Get initial observation
    obs = env.reset(seed=123)
    obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Encode
    with torch.no_grad():
        obs_embed = agent.encoder(obs_tensor)
        h, s = agent.rssm.initial_state(1)
        _, s, _, _ = agent.rssm.observe(h, s, torch.zeros(1, 1).to(device), obs_embed)
        h = agent.rssm.gru(torch.cat([s, torch.zeros(1, 1).to(device)], dim=-1), h)

    # Generate imagined trajectory
    imagined_frames = []
    real_frames = [obs]

    env_copy = SimplePendulum()
    env_copy.theta = env.theta
    env_copy.theta_dot = env.theta_dot

    for _ in range(20):
        with torch.no_grad():
            state = agent.get_state(h, s)
            action_dist = agent.actor(state)
            action = torch.tanh(action_dist.mean)

            # Decode current state
            recon = agent.decoder(state)
            imagined_frames.append(recon.squeeze().permute(1, 2, 0).cpu().numpy())

            # Step dynamics
            h, s, _ = agent.rssm(h, s, action)

        # Real environment
        real_obs, _, _, _ = env_copy.step(action.cpu().numpy().flatten()[0])
        real_frames.append(real_obs)

    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    indices = [0, 4, 9, 14, 19]

    for i, idx in enumerate(indices):
        axes[0, i].imshow(real_frames[idx])
        axes[0, i].set_title(f"Real (t={idx})")
        axes[0, i].axis("off")

        axes[1, i].imshow(imagined_frames[idx])
        axes[1, i].set_title(f"Imagined (t={idx})")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Imagined", fontsize=12)

    plt.suptitle("Real vs Imagined Trajectories", fontsize=14)
    plt.tight_layout()
    plt.savefig("dreamer_imagination.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'dreamer_imagination.png'")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Train
    agent, env, rewards_history = train_dreamer(num_episodes=150)

    # Visualize
    visualize_dreamer(agent, env, rewards_history)
    visualize_imagination(agent, env)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(
        """
Dreamer Architecture:
1. World Model (RSSM):
   - Deterministic state h: Long-term memory (GRU)
   - Stochastic state s: Captures uncertainty
   - Prior p(s|h): Prediction from dynamics only
   - Posterior q(s|h,o): Inference with observation
   
2. Actor-Critic:
   - Trained entirely on IMAGINED trajectories
   - No environment interaction during policy learning
   - Uses λ-returns for temporal credit assignment

Key Innovations:
- Latent imagination: Plan without rendering pixels
- Stochastic latents: Model uncertainty in dynamics  
- KL balancing: Prevent posterior collapse
- λ-returns: Better credit assignment than 1-step TD

Sample Efficiency:
- Dreamer matches model-free methods with 50x fewer samples
- DreamerV3 (2023) masters Minecraft with no human demonstrations

World Model Loss:
L = E[reconstruction + reward + KL(posterior || prior)]

Actor Objective:
Maximize: E[Σ γ^t r_t] estimated via imagined rollouts
    """
    )

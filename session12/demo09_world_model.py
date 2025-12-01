"""
Demo 09: World Models - Learning Visual Dynamics
================================================

This demo implements a simplified version of the World Models architecture
from Ha & Schmidhuber (2018), showing how agents can learn predictive
models of visual environments and train policies in "imagination."

Key Concepts:
- VAE (Vision): Compress high-dimensional images to latent codes
- MDN-RNN (Memory): Predict future latent states given actions
- Controller: Simple policy in latent space
- Dream training: Train policy on imagined rollouts

Architecture:
    Image -> VAE Encoder -> z (latent) -> MDN-RNN -> ẑ (predicted)
                                 |
                            Controller -> action

Historical Context:
- World Models (2018): First to show "dreaming" for visual RL
- Key insight: Learn dynamics model, train controller in imagination
- Dramatically more sample efficient than model-free methods
- Inspired Dreamer (2020) and DreamerV2/V3

Reference: Ha & Schmidhuber, "World Models" (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
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
# Simple Visual Environment
# ============================================================================


class SimpleCarEnv:
    """
    Simplified car environment inspired by CarRacing.

    - Top-down view of a car on a track
    - Actions: steer left/right, accelerate
    - Goal: Stay on track and move forward

    Visual observation: 64x64 RGB image
    """

    def __init__(self, img_size=64):
        self.img_size = img_size
        self.action_dim = 3  # [steer, accelerate, brake]

        # Car state
        self.x = 0.0  # position
        self.y = 0.0
        self.angle = 0.0  # heading
        self.speed = 0.0

        # Track parameters (simple oval)
        self.track_center = (32, 32)
        self.track_radius = 25
        self.track_width = 8

        self.max_steps = 100
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Start at random position on track
        angle = np.random.uniform(0, 2 * np.pi)
        self.x = self.track_center[0] + self.track_radius * np.cos(angle)
        self.y = self.track_center[1] + self.track_radius * np.sin(angle)
        self.angle = angle + np.pi / 2  # Tangent to track
        self.speed = 0.0
        self.steps = 0

        return self._render(), {}

    def step(self, action):
        self.steps += 1

        # Parse action
        if isinstance(action, (int, np.integer)):
            # Discrete action
            steer = [-0.3, 0, 0.3][action % 3]
            accel = 0.5
        else:
            # Continuous action
            steer = np.clip(action[0], -1, 1) * 0.3
            accel = np.clip(action[1], 0, 1)

        # Update car state
        self.angle += steer
        self.speed = self.speed * 0.95 + accel * 0.5  # Friction + acceleration
        self.speed = np.clip(self.speed, 0, 2)

        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)

        # Keep in bounds
        self.x = np.clip(self.x, 0, self.img_size - 1)
        self.y = np.clip(self.y, 0, self.img_size - 1)

        # Calculate reward
        dist_from_track = np.sqrt(
            (self.x - self.track_center[0]) ** 2 + (self.y - self.track_center[1]) ** 2
        )
        on_track = abs(dist_from_track - self.track_radius) < self.track_width

        if on_track:
            reward = self.speed * 0.1  # Reward for moving on track
        else:
            reward = -0.5  # Penalty for off track

        done = self.steps >= self.max_steps

        return self._render(), reward, done, False, {}

    def _render(self):
        """Render the environment."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Draw track (gray ring)
        for angle in np.linspace(0, 2 * np.pi, 100):
            for r in range(
                self.track_radius - self.track_width // 2,
                self.track_radius + self.track_width // 2,
            ):
                px = int(self.track_center[0] + r * np.cos(angle))
                py = int(self.track_center[1] + r * np.sin(angle))
                if 0 <= px < self.img_size and 0 <= py < self.img_size:
                    img[py, px] = [100, 100, 100]

        # Draw car (red rectangle)
        car_len = 4
        car_width = 2
        cx, cy = int(self.x), int(self.y)

        for dx in range(-car_len, car_len + 1):
            for dy in range(-car_width, car_width + 1):
                px = int(cx + dx * np.cos(self.angle) - dy * np.sin(self.angle))
                py = int(cy + dx * np.sin(self.angle) + dy * np.cos(self.angle))
                if 0 <= px < self.img_size and 0 <= py < self.img_size:
                    img[py, px] = [255, 0, 0]

        return img


# ============================================================================
# Variational Autoencoder (VAE) - The Vision Component
# ============================================================================


class VAE(nn.Module):
    """
    Variational Autoencoder for compressing visual observations.

    The VAE learns a compressed latent representation z that captures
    the essential information in the image. This is the "Vision" component
    of World Models.

    Key idea: Learn p(z|x) and p(x|z) such that z is low-dimensional
    but contains enough info to reconstruct x.
    """

    def __init__(self, latent_dim=32, img_channels=3, img_size=64):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # Encoder: image -> latent distribution parameters
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: latent -> reconstructed image
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode image to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to reconstructed image."""
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence

    - Reconstruction: How well can we rebuild the image?
    - KL: Keep latent distribution close to standard normal
    """
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ============================================================================
# MDN-RNN - The Memory/Dynamics Component
# ============================================================================


class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN for predicting future latent states.

    Given current latent z_t and action a_t, predict distribution
    over next latent z_{t+1}.

    Uses mixture of Gaussians to capture multimodal futures.

    This is the "Memory" component that learns environment dynamics.
    """

    def __init__(
        self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5, n_layers=1
    ):
        super(MDNRNN, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # MDN output heads
        # For each Gaussian: mu (latent_dim), sigma (latent_dim), and mixing weight
        self.fc_pi = nn.Linear(hidden_dim, n_gaussians)  # Mixing coefficients
        self.fc_mu = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, n_gaussians * latent_dim)

        # Also predict reward
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden=None):
        """
        Forward pass: predict next latent distribution.

        Args:
            z: (batch, seq, latent_dim) latent states
            action: (batch, seq, action_dim) actions
            hidden: LSTM hidden state

        Returns:
            pi: Mixture weights
            mu: Means of Gaussians
            sigma: Std devs of Gaussians
            reward: Predicted reward
            hidden: Updated LSTM hidden state
        """
        # Concatenate latent and action
        x = torch.cat([z, action], dim=-1)

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # MDN outputs
        pi = F.softmax(self.fc_pi(lstm_out), dim=-1)  # (batch, seq, n_gaussians)
        mu = self.fc_mu(lstm_out).view(
            *lstm_out.shape[:-1], self.n_gaussians, self.latent_dim
        )
        sigma = F.softplus(self.fc_sigma(lstm_out)).view(
            *lstm_out.shape[:-1], self.n_gaussians, self.latent_dim
        )
        sigma = sigma + 1e-6  # Numerical stability

        reward = self.fc_reward(lstm_out)

        return pi, mu, sigma, reward, hidden

    def sample(self, pi, mu, sigma):
        """Sample from the mixture of Gaussians."""
        batch_shape = pi.shape[:-1]

        # Sample which Gaussian
        idx = Categorical(pi).sample()  # (batch,) or (batch, seq)

        # Gather corresponding mu and sigma
        idx_expanded = (
            idx.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, self.latent_dim)
        )
        mu_selected = mu.gather(-2, idx_expanded).squeeze(-2)
        sigma_selected = sigma.gather(-2, idx_expanded).squeeze(-2)

        # Sample from selected Gaussian
        z_next = mu_selected + sigma_selected * torch.randn_like(sigma_selected)

        return z_next


def mdn_loss(pi, mu, sigma, target):
    """
    Negative log-likelihood of target under mixture of Gaussians.

    Args:
        pi: (batch, seq, n_gaussians) mixing coefficients
        mu: (batch, seq, n_gaussians, latent_dim) means
        sigma: (batch, seq, n_gaussians, latent_dim) std devs
        target: (batch, seq, latent_dim) target latent states
    """
    # Expand target for broadcasting
    target = target.unsqueeze(-2)  # (batch, seq, 1, latent_dim)

    # Log probability under each Gaussian
    log_prob = (
        -0.5 * ((target - mu) / sigma).pow(2)
        - torch.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )
    log_prob = log_prob.sum(dim=-1)  # Sum over latent dimensions

    # Log-sum-exp with mixing coefficients
    log_pi = torch.log(pi + 1e-10)
    log_prob = torch.logsumexp(log_pi + log_prob, dim=-1)

    return -log_prob.mean()


# ============================================================================
# Simple Controller
# ============================================================================


class Controller(nn.Module):
    """
    Simple linear controller that maps [z, h] to actions.

    In the original World Models paper, this is trained via
    evolutionary strategies (CMA-ES) on imagined rollouts.

    Here we use a simple policy gradient for demonstration.
    """

    def __init__(self, latent_dim, hidden_dim, action_dim):
        super(Controller, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, z, h):
        """
        Args:
            z: (batch, latent_dim) current latent
            h: (batch, hidden_dim) LSTM hidden state
        """
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)


# ============================================================================
# Training Functions
# ============================================================================


def collect_data(env, num_episodes=100, max_steps=100):
    """Collect random rollout data from environment."""
    print("Collecting random rollout data...")

    observations = []
    actions = []
    rewards = []

    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=SEED + ep)
        ep_obs = [obs]
        ep_actions = []
        ep_rewards = []

        for step in range(max_steps):
            action = np.random.randint(0, 3)  # Random action
            obs, reward, done, _, _ = env.step(action)

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)

            if done:
                break

        observations.append(ep_obs)
        actions.append(ep_actions)
        rewards.append(ep_rewards)

    return observations, actions, rewards


def train_vae(vae, observations, epochs=20, batch_size=32, lr=1e-3):
    """Train the VAE on collected observations."""
    print("\nTraining VAE...")

    # Flatten observations and convert to tensor
    all_obs = []
    for ep_obs in observations:
        all_obs.extend(ep_obs)

    all_obs = np.array(all_obs).astype(np.float32) / 255.0
    all_obs = torch.FloatTensor(all_obs).permute(0, 3, 1, 2).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(all_obs))
        total_loss = 0

        for i in range(0, len(all_obs), batch_size):
            batch_idx = perm[i : i + batch_size]
            batch = all_obs[batch_idx]

            recon, mu, logvar, z = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(all_obs) / batch_size)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.2f}")

    return losses


def train_mdnrnn(
    vae,
    mdnrnn,
    observations,
    actions,
    rewards,
    epochs=20,
    batch_size=16,
    seq_len=20,
    lr=1e-3,
):
    """Train the MDN-RNN on latent sequences."""
    print("\nTraining MDN-RNN...")

    optimizer = optim.Adam(mdnrnn.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        # Sample random sequences
        for _ in range(len(observations) // batch_size):
            # Sample episodes
            ep_indices = np.random.choice(len(observations), batch_size)

            batch_z = []
            batch_actions = []
            batch_z_next = []

            for ep_idx in ep_indices:
                ep_obs = observations[ep_idx]
                ep_act = actions[ep_idx]

                if len(ep_obs) <= seq_len:
                    continue

                # Random starting point
                start = np.random.randint(0, len(ep_obs) - seq_len)

                # Get observations and encode to latent
                obs_seq = (
                    np.array(ep_obs[start : start + seq_len + 1]).astype(np.float32)
                    / 255.0
                )
                obs_tensor = torch.FloatTensor(obs_seq).permute(0, 3, 1, 2).to(device)

                with torch.no_grad():
                    mu, _ = vae.encode(obs_tensor)
                    z_seq = mu  # Use mean for deterministic encoding

                # Actions (one-hot)
                act_seq = ep_act[start : start + seq_len]
                act_onehot = np.eye(3)[act_seq]  # One-hot encode

                batch_z.append(z_seq[:-1])
                batch_actions.append(act_onehot)
                batch_z_next.append(z_seq[1:])

            if len(batch_z) < 2:
                continue

            # Stack batch
            z = torch.stack(batch_z)
            act = torch.FloatTensor(np.array(batch_actions)).to(device)
            z_next = torch.stack(batch_z_next)

            # Forward
            pi, mu, sigma, _, _ = mdnrnn(z, act)

            # Loss
            loss = mdn_loss(pi, mu, sigma, z_next)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = total_loss / n_batches
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    return losses


# ============================================================================
# Visualization
# ============================================================================


def visualize_world_model(vae, mdnrnn, env, observations):
    """Visualize world model components."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. VAE reconstruction
    ax1 = axes[0, 0]

    # Get a sample observation
    sample_obs = observations[0][10]
    sample_tensor = (
        torch.FloatTensor(sample_obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    with torch.no_grad():
        recon, _, _, z = vae(sample_tensor)

    ax1_left = ax1
    ax1_left.imshow(sample_obs)
    ax1_left.set_title("Original")
    ax1_left.axis("off")

    # 2. Reconstruction
    ax2 = axes[0, 1]
    recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()
    ax2.imshow(recon_img)
    ax2.set_title("VAE Reconstruction")
    ax2.axis("off")

    # 3. Latent space visualization
    ax3 = axes[0, 2]

    # Encode multiple observations
    all_z = []
    all_angles = []

    for ep_idx in range(min(10, len(observations))):
        for obs in observations[ep_idx][::5]:  # Every 5th frame
            obs_tensor = (
                torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )
            with torch.no_grad():
                mu, _ = vae.encode(obs_tensor)
            all_z.append(mu.cpu().numpy().flatten())

    all_z = np.array(all_z)

    # Plot first two latent dimensions
    scatter = ax3.scatter(
        all_z[:, 0], all_z[:, 1], c=range(len(all_z)), cmap="viridis", alpha=0.5, s=10
    )
    ax3.set_xlabel("z[0]")
    ax3.set_ylabel("z[1]")
    ax3.set_title("Latent Space (first 2 dims)")
    plt.colorbar(scatter, ax=ax3, label="Time")

    # 4-6. Dream sequence (imagined rollout)
    # Start from a real observation
    start_obs = observations[5][0]
    start_tensor = (
        torch.FloatTensor(start_obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    with torch.no_grad():
        mu, _ = vae.encode(start_tensor)
        z = mu

    # Generate dream sequence
    dream_frames = [vae.decode(z).squeeze().permute(1, 2, 0).cpu().numpy()]
    hidden = None

    for step in range(10):
        # Random action
        action = torch.zeros(1, 1, 3).to(device)
        action[0, 0, 1] = 1.0  # Forward

        with torch.no_grad():
            pi, mu, sigma, _, hidden = mdnrnn(z.unsqueeze(1), action, hidden)
            z = mdnrnn.sample(pi.squeeze(1), mu.squeeze(1), sigma.squeeze(1))

            # Decode to image
            dream_img = vae.decode(z).squeeze().permute(1, 2, 0).cpu().numpy()
            dream_frames.append(dream_img)

    # Show dream frames
    for i, (ax, idx) in enumerate(
        zip([axes[1, 0], axes[1, 1], axes[1, 2]], [0, 5, 10])
    ):
        if idx < len(dream_frames):
            ax.imshow(dream_frames[idx])
            ax.set_title(f"Dream Step {idx}")
        ax.axis("off")

    fig.suptitle("World Model Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig("world_model_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to 'world_model_results.png'")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("Demo 09: World Models - Learning Visual Dynamics")
    print("=" * 60)
    print()

    # Create environment
    env = SimpleCarEnv(img_size=64)
    print(f"Environment: SimpleCarEnv (64x64 RGB)")
    print(f"Action space: 3 (discrete)")
    print()

    # Collect data
    observations, actions, rewards = collect_data(env, num_episodes=50, max_steps=50)
    print(f"Collected {sum(len(ep) for ep in observations)} frames")

    # Create models
    latent_dim = 32
    vae = VAE(latent_dim=latent_dim, img_channels=3, img_size=64).to(device)
    mdnrnn = MDNRNN(
        latent_dim=latent_dim, action_dim=3, hidden_dim=128, n_gaussians=3
    ).to(device)

    print(f"\nVAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"MDN-RNN parameters: {sum(p.numel() for p in mdnrnn.parameters()):,}")

    # Train VAE
    vae_losses = train_vae(vae, observations, epochs=30, batch_size=32)

    # Train MDN-RNN
    rnn_losses = train_mdnrnn(
        vae, mdnrnn, observations, actions, rewards, epochs=30, batch_size=8, seq_len=15
    )

    # Visualize
    visualize_world_model(vae, mdnrnn, env, observations)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(vae_losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("VAE Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rnn_losses)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("MDN-RNN Training Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("world_model_training.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(
        """
1. VAE compresses 64x64x3 images to 32-dimensional latent codes
2. MDN-RNN predicts distribution over next latent states
3. Together they form a "world model" that can imagine futures
4. Controller can be trained on imagined rollouts ("dreaming")
5. Much more sample efficient than model-free RL

Key equations:
- VAE: z ~ q(z|x), x' ~ p(x|z)
- MDN-RNN: p(z_{t+1}|z_t, a_t, h_t) = Σ π_k N(μ_k, σ_k)
- Training: Maximize ELBO for VAE, minimize NLL for MDN

This architecture inspired:
- Dreamer (2020): Actor-critic in latent space
- DreamerV2 (2021): Discrete latents
- DreamerV3 (2023): Cross-domain generalization
    """
    )


if __name__ == "__main__":
    print(__doc__)
    main()

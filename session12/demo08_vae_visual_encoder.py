"""
Demo 08: Variational Autoencoder (VAE) for Visual Encoding
==========================================================

This demo implements a VAE for learning compressed representations
of visual observations - the "Vision" component of World Models.

Key Concepts:
- Encoder: Image -> latent distribution (μ, σ)
- Reparameterization trick: z = μ + σ * ε (enables backprop through sampling)
- Decoder: Latent z -> reconstructed image
- Loss = Reconstruction + KL divergence

Why VAE for RL?
- Compress high-dimensional images to low-dimensional latent codes
- Latent space is smooth and interpolatable
- Can generate/imagine new observations
- Provides compact state representation for RL

Historical Context:
- VAE introduced by Kingma & Welling (2014)
- Used in World Models (Ha & Schmidhuber, 2018) for visual encoding
- Foundation for latent space world models (Dreamer, etc.)

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from sklearn.decomposition import PCA
import cv2

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Generate Synthetic Visual Dataset
# ============================================================================


def generate_shape_dataset(n_samples=5000, img_size=64):
    """
    Generate synthetic dataset of simple shapes.

    Each image contains a colored shape (circle, square, triangle)
    at random positions. This simulates visual observations from an environment.
    """
    images = []
    labels = []  # For visualization (shape type, position, color)

    for i in range(n_samples):
        # Create blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Random shape type
        shape_type = np.random.randint(0, 3)  # 0: circle, 1: square, 2: triangle

        # Random position
        cx = np.random.randint(15, img_size - 15)
        cy = np.random.randint(15, img_size - 15)

        # Random size
        size = np.random.randint(8, 15)

        # Random color (excluding very dark colors)
        color = tuple(np.random.randint(100, 256, 3).tolist())

        if shape_type == 0:  # Circle
            cv2.circle(img, (cx, cy), size, color, -1)
        elif shape_type == 1:  # Square
            cv2.rectangle(
                img, (cx - size, cy - size), (cx + size, cy + size), color, -1
            )
        else:  # Triangle
            pts = np.array(
                [[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]],
                np.int32,
            )
            cv2.fillPoly(img, [pts], color)

        images.append(img)
        labels.append(
            {"shape": shape_type, "position": (cx, cy), "size": size, "color": color}
        )

    return np.array(images), labels


# ============================================================================
# VAE Architecture
# ============================================================================


class VAE(nn.Module):
    """
    Variational Autoencoder for visual observations.

    Architecture:
    - Encoder: Conv layers -> flatten -> fc -> (μ, log σ²)
    - Decoder: fc -> reshape -> deconv layers -> image
    """

    def __init__(self, img_channels=3, img_size=64, latent_dim=32):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder
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

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x):
        """Encode image to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε

        This allows gradients to flow through the sampling operation.
        Instead of sampling z ~ N(μ, σ²), we sample ε ~ N(0, 1)
        and compute z = μ + σ * ε.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent code to image."""
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence

    Reconstruction: How well can we rebuild the input?
    KL Divergence: How close is q(z|x) to the prior p(z) = N(0, I)?

    β-VAE: β > 1 encourages more disentangled representations
    """
    # Reconstruction loss (per pixel MSE)
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ============================================================================
# Training
# ============================================================================


def train_vae(vae, train_loader, epochs=30, lr=1e-3, beta=1.0):
    """Train the VAE."""
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(epochs):
        vae.train()
        epoch_total = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)

            recon, mu, logvar, z = vae(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        history["total_loss"].append(epoch_total / n_batches)
        history["recon_loss"].append(epoch_recon / n_batches)
        history["kl_loss"].append(epoch_kl / n_batches)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Total={epoch_total/n_batches:.2f}, "
                f"Recon={epoch_recon/n_batches:.2f}, "
                f"KL={epoch_kl/n_batches:.2f}"
            )

    return history


# ============================================================================
# Visualization
# ============================================================================


def visualize_reconstructions(vae, images, n_samples=8):
    """Show original and reconstructed images."""
    vae.eval()

    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))

    indices = np.random.choice(len(images), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img = images[idx]
            img_tensor = (
                torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )

            recon, _, _, _ = vae(img_tensor)
            recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()

            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_ylabel("Original", fontsize=12)

            axes[1, i].imshow(recon_img)
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_ylabel("Reconstructed", fontsize=12)

    plt.suptitle("VAE Reconstruction Quality", fontsize=14)
    plt.tight_layout()
    plt.savefig("vae_reconstructions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'vae_reconstructions.png'")


def visualize_latent_space(vae, images, labels, n_samples=1000):
    """Visualize the learned latent space."""
    vae.eval()

    # Encode all images
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)

    all_z = []
    all_shapes = []
    all_positions = []

    with torch.no_grad():
        for idx in indices:
            img = images[idx]
            img_tensor = (
                torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )

            mu, _ = vae.encode(img_tensor)
            all_z.append(mu.cpu().numpy().flatten())
            all_shapes.append(labels[idx]["shape"])
            all_positions.append(labels[idx]["position"])

    all_z = np.array(all_z)
    all_shapes = np.array(all_shapes)
    all_positions = np.array(all_positions)

    # Use PCA to visualize high-dimensional latent space in 2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Color by shape type
    ax1 = axes[0]
    shape_names = ["Circle", "Square", "Triangle"]
    colors = ["red", "blue", "green"]

    for shape in range(3):
        mask = all_shapes == shape
        ax1.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            c=colors[shape],
            label=shape_names[shape],
            alpha=0.5,
            s=20,
        )

    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_title("Latent Space by Shape Type")
    ax1.legend()

    # 2. Color by X position
    ax2 = axes[1]
    scatter = ax2.scatter(
        z_2d[:, 0], z_2d[:, 1], c=all_positions[:, 0], cmap="viridis", alpha=0.5, s=20
    )
    plt.colorbar(scatter, ax=ax2, label="X position")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.set_title("Latent Space by X Position")

    # 3. Color by Y position
    ax3 = axes[2]
    scatter = ax3.scatter(
        z_2d[:, 0], z_2d[:, 1], c=all_positions[:, 1], cmap="plasma", alpha=0.5, s=20
    )
    plt.colorbar(scatter, ax=ax3, label="Y position")
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.set_title("Latent Space by Y Position")

    plt.tight_layout()
    plt.savefig("vae_latent_space.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'vae_latent_space.png'")

    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")


def visualize_latent_traversal(vae, images):
    """
    Traverse individual latent dimensions to see what they encode.
    This reveals what each dimension "controls".
    """
    vae.eval()

    # Pick a reference image
    ref_img = images[0]
    ref_tensor = (
        torch.FloatTensor(ref_img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    with torch.no_grad():
        mu, _ = vae.encode(ref_tensor)

    # Traverse first 4 latent dimensions
    n_dims = 4
    n_steps = 7

    fig, axes = plt.subplots(n_dims, n_steps, figsize=(2 * n_steps, 2 * n_dims))

    values = np.linspace(-3, 3, n_steps)

    for dim in range(n_dims):
        for i, val in enumerate(values):
            z = mu.clone()
            z[0, dim] = val

            with torch.no_grad():
                recon = vae.decode(z)

            img = recon.squeeze().permute(1, 2, 0).cpu().numpy()

            axes[dim, i].imshow(img)
            axes[dim, i].axis("off")

            if i == 0:
                axes[dim, i].set_ylabel(f"z[{dim}]", fontsize=12)
            if dim == 0:
                axes[dim, i].set_title(f"{val:.1f}", fontsize=10)

    plt.suptitle(
        "Latent Dimension Traversal\n(Each row varies one latent dimension)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("vae_latent_traversal.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'vae_latent_traversal.png'")


def visualize_interpolation(vae, images):
    """Interpolate between two images in latent space."""
    vae.eval()

    # Pick two different images
    idx1, idx2 = 0, 100
    img1 = images[idx1]
    img2 = images[idx2]

    img1_tensor = (
        torch.FloatTensor(img1).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )
    img2_tensor = (
        torch.FloatTensor(img2).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    with torch.no_grad():
        mu1, _ = vae.encode(img1_tensor)
        mu2, _ = vae.encode(img2_tensor)

    # Interpolate
    n_steps = 10
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))

    for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
        z = (1 - alpha) * mu1 + alpha * mu2

        with torch.no_grad():
            recon = vae.decode(z)

        img = recon.squeeze().permute(1, 2, 0).cpu().numpy()

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"α={alpha:.1f}", fontsize=10)

    plt.suptitle(
        "Latent Space Interpolation\n(Smooth transition between two images)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("vae_interpolation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'vae_interpolation.png'")


def visualize_training_history(history):
    """Plot training losses."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1 = axes[0]
    ax1.plot(history["total_loss"], "b-", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total VAE Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history["recon_loss"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Reconstruction Loss")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(history["kl_loss"], "r-", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("KL Divergence")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vae_training.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 'vae_training.png'")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("Demo 08: VAE Visual Encoder")
    print("=" * 60)
    print()

    # Generate dataset
    print("Generating synthetic visual dataset...")
    images, labels = generate_shape_dataset(n_samples=5000, img_size=64)
    print(f"Generated {len(images)} images of shape {images[0].shape}")

    # Prepare data
    images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
    dataset = TensorDataset(images_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create and train VAE
    latent_dim = 32
    vae = VAE(img_channels=3, img_size=64, latent_dim=latent_dim).to(device)
    print(f"\nVAE architecture:")
    print(f"  Input: 64x64x3 image")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Parameters: {sum(p.numel() for p in vae.parameters()):,}")

    print("\nTraining VAE...")
    history = train_vae(vae, train_loader, epochs=30, lr=1e-3, beta=1.0)

    # Visualizations
    print("\nGenerating visualizations...")

    visualize_training_history(history)
    visualize_reconstructions(vae, images)
    visualize_latent_space(vae, images, labels)
    visualize_latent_traversal(vae, images)
    visualize_interpolation(vae, images)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print(f"""
1. VAE learns to compress {64*64*3}-dim images to {latent_dim}-dim latent codes
2. The latent space is smooth (nearby points = similar images)
3. Can generate new images by sampling z ~ N(0, I)
4. Reparameterization trick enables backprop through sampling
5. KL term regularizes latent space to be Gaussian

VAE Loss = Reconstruction + β * KL
- Reconstruction: E[log p(x|z)] ≈ -MSE(x, x̂)
- KL: KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log σ² - μ² - σ²)

Applications in Visual RL:
- World Models: VAE encodes observations for dynamics model
- Dreamer: Learns in latent space for efficiency
- Goal-conditioned RL: Goals specified in latent space
- Model-based RL: Planning in compressed state space
    """)


if __name__ == "__main__":
    print(__doc__)
    main()

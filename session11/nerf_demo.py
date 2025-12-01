"""
NeRF (Neural Radiance Fields) Demo
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements a simplified NeRF for novel view synthesis.
Key concepts:
- Implicit neural scene representation
- Positional encoding for high-frequency details
- Volume rendering via ray marching
- Hierarchical sampling for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# Positional Encoding
# ============================================================================


class PositionalEncoder:
    """
    Positional encoding to map inputs to higher dimensional space.

    gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ...,
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

    This enables the MLP to learn high-frequency functions.
    """

    def __init__(self, L):
        """
        Args:
            L: Number of frequency bands
        """
        self.L = L
        self.freq_bands = 2.0 ** torch.linspace(0, L - 1, L)

    def encode(self, x):
        """
        Args:
            x: [..., C] - Input coordinates
        Returns:
            [..., C * 2L] - Encoded coordinates
        """
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)

    def output_dim(self, input_dim):
        return input_dim * 2 * self.L


# ============================================================================
# NeRF MLP Network
# ============================================================================


class NeRF(nn.Module):
    """
    NeRF MLP that maps (position, direction) to (color, density).

    Architecture:
    - 8 FC layers (256 units) for processing position
    - Skip connection at layer 4
    - Separate branches for density and color
    - View direction conditioning for color
    """

    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256):
        super(NeRF, self).__init__()

        # Positional encoders
        self.pos_encoder = PositionalEncoder(pos_L)
        self.dir_encoder = PositionalEncoder(dir_L)

        pos_dim = self.pos_encoder.output_dim(3)  # 3D position
        dir_dim = self.dir_encoder.output_dim(3)  # 3D direction

        # Position processing layers (8 layers with skip at 4)
        self.pos_layers1 = nn.ModuleList(
            [
                nn.Linear(pos_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        # Skip connection layer
        self.pos_layers2 = nn.ModuleList(
            [
                nn.Linear(hidden_dim + pos_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        # Density head (single layer)
        self.density_head = nn.Linear(hidden_dim, 1)

        # Feature for color prediction
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        # Color head (conditioned on direction)
        self.color_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, 3),  # RGB
            ]
        )

    def forward(self, pos, view_dir):
        """
        Args:
            pos: [N, 3] - 3D positions
            view_dir: [N, 3] - Viewing directions (unit vectors)
        Returns:
            rgb: [N, 3] - RGB colors
            sigma: [N, 1] - Volume densities
        """
        # Encode inputs
        pos_enc = self.pos_encoder.encode(pos)
        dir_enc = self.dir_encoder.encode(view_dir)

        # Process position
        x = pos_enc
        for layer in self.pos_layers1:
            x = F.relu(layer(x))

        # Skip connection
        x = torch.cat([x, pos_enc], dim=-1)
        for layer in self.pos_layers2:
            x = F.relu(layer(x))

        # Density (depends only on position)
        sigma = F.relu(self.density_head(x))

        # Feature for color
        feat = self.feature_layer(x)

        # Color (depends on position and direction)
        x = torch.cat([feat, dir_enc], dim=-1)
        for i, layer in enumerate(self.color_layers):
            x = layer(x)
            if i < len(self.color_layers) - 1:
                x = F.relu(x)

        rgb = torch.sigmoid(x)  # RGB in [0, 1]

        return rgb, sigma


# ============================================================================
# Ray Sampling and Volume Rendering
# ============================================================================


def get_rays(H, W, focal, c2w):
    """
    Generate rays for all pixels in an image.

    Args:
        H, W: Image height and width
        focal: Focal length
        c2w: [4, 4] Camera-to-world transformation matrix
    Returns:
        rays_o: [H*W, 3] Ray origins
        rays_d: [H*W, 3] Ray directions (unit vectors)
    """
    # Pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )

    # Normalized device coordinates
    dirs = torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], dim=-1
    )

    # Rotate ray directions from camera to world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    # Ray origins (camera center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def sample_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    """
    Sample points along rays.

    Args:
        rays_o: [N_rays, 3] Ray origins
        rays_d: [N_rays, 3] Ray directions
        near, far: Near and far bounds
        N_samples: Number of samples per ray
        perturb: Whether to add random perturbations
    Returns:
        pts: [N_rays, N_samples, 3] Sampled 3D points
        z_vals: [N_rays, N_samples] Depth values
    """
    N_rays = rays_o.shape[0]

    # Linearly spaced samples
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(N_rays, N_samples)

    # Stratified sampling with perturbation
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    # Points along rays: r(t) = o + t*d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


def volume_render(rgb, sigma, z_vals, rays_d, noise_std=0.0):
    """
    Volume rendering using the classic equation.

    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
    T_i = exp(-sum_{j<i} sigma_j * delta_j)

    Args:
        rgb: [N_rays, N_samples, 3] RGB values
        sigma: [N_rays, N_samples, 1] Densities
        z_vals: [N_rays, N_samples] Depth values
        rays_d: [N_rays, 3] Ray directions
        noise_std: Std of noise to add to sigma (regularization)
    Returns:
        rgb_map: [N_rays, 3] Rendered RGB
        depth_map: [N_rays] Rendered depth
        acc_map: [N_rays] Accumulated opacity
        weights: [N_rays, N_samples] Weights for each sample
    """
    # Distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)],
        dim=-1,
    )

    # Multiply distances by ray direction norm (accounts for non-unit directions)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to sigma (for regularization during training)
    if noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * noise_std

    # Compute alpha values
    alpha = 1.0 - torch.exp(-F.relu(sigma[..., 0]) * dists)

    # Compute transmittance T_i
    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]

    # Compute weights
    weights = alpha * transmittance

    # Render RGB
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # Render depth
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map, weights


# ============================================================================
# Synthetic Dataset
# ============================================================================


class SyntheticDataset:
    """
    Synthetic dataset for NeRF training.
    In practice, you would load actual images and camera poses.
    """

    def __init__(self, num_images=100, H=100, W=100, focal=100):
        self.num_images = num_images
        self.H = H
        self.W = W
        self.focal = focal

        # Generate synthetic images and poses
        self.images, self.poses = self._generate_data()

    def _generate_data(self):
        """Generate simple synthetic data."""
        images = []
        poses = []

        for i in range(self.num_images):
            # Camera pose (circular trajectory)
            theta = 2 * np.pi * i / self.num_images
            pose = torch.tensor(
                [
                    [np.cos(theta), 0, np.sin(theta), 3 * np.sin(theta)],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 3 * np.cos(theta)],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )

            # Simple synthetic image (gradient)
            img = torch.zeros((self.H, self.W, 3), dtype=torch.float32)
            img[..., 0] = torch.linspace(0, 1, self.H).unsqueeze(1).repeat(1, self.W)
            img[..., 1] = torch.linspace(0, 1, self.W).unsqueeze(0).repeat(self.H, 1)
            img[..., 2] = 0.5

            images.append(img)
            poses.append(pose)

        return images, poses

    def get_batch(self, batch_size=1024):
        """Get random batch of rays."""
        # Random image
        img_idx = np.random.randint(0, self.num_images)
        img = self.images[img_idx]
        pose = self.poses[img_idx]

        # Generate rays for this image
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose)

        # Flatten
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target = img.reshape(-1, 3)

        # Random sample
        indices = torch.randint(0, rays_o.shape[0], (batch_size,))

        return rays_o[indices], rays_d[indices], target[indices]


# ============================================================================
# Training Function
# ============================================================================


def train_nerf(args):
    """Train NeRF model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticDataset(
        num_images=args.num_images,
        H=args.img_size,
        W=args.img_size,
        focal=args.img_size,
    )
    print(f"Number of training images: {args.num_images}")
    print(f"Image resolution: {args.img_size}x{args.img_size}")

    # Create model
    print("\nInitializing NeRF...")
    model = NeRF(pos_L=args.pos_L, dir_L=args.dir_L, hidden_dim=args.hidden_dim).to(
        device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params/1e6:.1f}M")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    # Training loop
    print("\nTraining...")
    model.train()

    pbar = tqdm(range(args.n_iters))
    for iter in pbar:
        # Get batch of rays
        rays_o, rays_d, target_rgb = dataset.get_batch(args.batch_size)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_rgb = target_rgb.to(device)

        # Sample points along rays
        pts, z_vals = sample_along_rays(
            rays_o,
            rays_d,
            near=args.near,
            far=args.far,
            N_samples=args.n_samples,
            perturb=True,
        )

        # Flatten for network
        N_rays, N_samples, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand(N_rays, N_samples, 3).reshape(-1, 3)

        # Query network
        rgb, sigma = model(pts_flat, dirs_flat)

        # Reshape
        rgb = rgb.reshape(N_rays, N_samples, 3)
        sigma = sigma.reshape(N_rays, N_samples, 1)

        # Volume rendering
        rgb_map, depth_map, acc_map, weights = volume_render(
            rgb,
            sigma,
            z_vals,
            rays_d,
            noise_std=1.0 if iter < args.n_iters // 2 else 0.0,
        )

        # Loss
        loss = F.mse_loss(rgb_map, target_rgb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute PSNR
        with torch.no_grad():
            psnr = -10.0 * torch.log10(loss)

        # Update progress bar
        if iter % 100 == 0:
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "PSNR": f"{psnr.item():.2f} dB"}
            )

    print("\nTraining complete!")

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "nerf_model.pth",
    )
    print("Model saved to nerf_model.pth")

    return model, dataset


# ============================================================================
# Rendering Function
# ============================================================================


@torch.no_grad()
def render_image(
    model, pose, H, W, focal, near, far, N_samples, chunk=1024, device="cuda"
):
    """Render a full image from a camera pose."""
    model.eval()

    # Generate rays
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    # Render in chunks
    all_rgb = []
    all_depth = []

    for i in range(0, rays_o.shape[0], chunk):
        rays_o_chunk = rays_o[i : i + chunk]
        rays_d_chunk = rays_d[i : i + chunk]

        # Sample points
        pts, z_vals = sample_along_rays(
            rays_o_chunk, rays_d_chunk, near, far, N_samples, perturb=False
        )

        # Query network
        N_rays, N_samples, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d_chunk[:, None, :].expand(N_rays, N_samples, 3).reshape(-1, 3)

        rgb, sigma = model(pts_flat, dirs_flat)
        rgb = rgb.reshape(N_rays, N_samples, 3)
        sigma = sigma.reshape(N_rays, N_samples, 1)

        # Volume render
        rgb_map, depth_map, _, _ = volume_render(
            rgb, sigma, z_vals, rays_d_chunk, noise_std=0.0
        )

        all_rgb.append(rgb_map.cpu())
        all_depth.append(depth_map.cpu())

    # Concatenate
    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    depth = torch.cat(all_depth, dim=0).reshape(H, W)

    return rgb.numpy(), depth.numpy()


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("NeRF Demo for Novel View Synthesis")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    # Train
    model, dataset = train_nerf(args)

    # Render test views
    print("\nRendering test views...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Render from different viewpoints
    test_poses = []
    for i in range(5):
        theta = 2 * np.pi * (i / 5 + 0.1)  # Offset from training views
        pose = torch.tensor(
            [
                [np.cos(theta), 0, np.sin(theta), 3 * np.sin(theta)],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 3 * np.cos(theta)],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        test_poses.append(pose)

    # Render and display
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, pose in enumerate(test_poses):
        rgb, depth = render_image(
            model,
            pose,
            dataset.H,
            dataset.W,
            dataset.focal,
            args.near,
            args.far,
            args.n_samples,
            device=device,
        )

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"Novel View {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(depth, cmap="viridis")
        axes[1, i].set_title(f"Depth {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("nerf_results.png", dpi=150, bbox_inches="tight")
    print("Results saved to nerf_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRF Demo")

    # Dataset
    parser.add_argument("--scene", type=str, default="synthetic", help="Scene name")
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of training images"
    )
    parser.add_argument("--img_size", type=int, default=100, help="Image size")

    # Model
    parser.add_argument(
        "--pos_L",
        type=int,
        default=10,
        help="Positional encoding frequency bands for position",
    )
    parser.add_argument(
        "--dir_L",
        type=int,
        default=4,
        help="Positional encoding frequency bands for direction",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden layer dimension"
    )

    # Training
    parser.add_argument(
        "--n_iters", type=int, default=10000, help="Number of training iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (number of rays)"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Rendering
    parser.add_argument("--near", type=float, default=2.0, help="Near clipping plane")
    parser.add_argument("--far", type=float, default=6.0, help="Far clipping plane")
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of samples per ray"
    )

    args = parser.parse_args()

    main(args)

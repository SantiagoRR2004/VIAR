"""
NeRF (Neural Radiance Fields) Demo
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements a simplified NeRF for novel view synthesis
on a synthetic 3D scene (a colored sphere).

Key concepts:
- Implicit neural scene representation: F_Θ(x, d) -> (color, density)
- Positional encoding for high-frequency details
- Volume rendering via ray marching
- Training by matching rendered pixels to input images

NOTE: This demo uses a single NeRF (no coarse/fine hierarchical sampling)
with uniform stratified samples along each ray, to keep the code simple.
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
    Positional encoding to map inputs to higher-dimensional space.

    gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ...,
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]
    """

    def __init__(self, L):
        """
        Args:
            L: number of frequency bands
        """
        self.L = L

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., C] input coordinates (on CPU or CUDA)

        Returns:
            [..., C * 2L] encoded coordinates (same device/dtype as x)
        """
        device = x.device
        dtype = x.dtype

        # Build frequency bands on the fly on the correct device
        freq_bands = 2.0 ** torch.linspace(
            0, self.L - 1, self.L, device=device, dtype=dtype
        )

        out = []
        for freq in freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))

        return torch.cat(out, dim=-1)

    def output_dim(self, input_dim: int) -> int:
        return input_dim * 2 * self.L


# ============================================================================
# NeRF MLP Network
# ============================================================================


class NeRF(nn.Module):
    """
    NeRF MLP that maps (position, direction) to (color, density).

    Architecture (very close to original NeRF):
    - Positional encoding for 3D position and viewing direction
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

        # Position processing layers (first 4 layers)
        self.pos_layers1 = nn.ModuleList(
            [
                nn.Linear(pos_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        # Position processing layers (last 4 layers, with skip)
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

    def forward(self, pos: torch.Tensor, view_dir: torch.Tensor):
        """
        Args:
            pos: [N, 3] - 3D positions
            view_dir: [N, 3] - Viewing directions (unit vectors)
        Returns:
            rgb: [N, 3] - RGB colors in [0,1]
            sigma: [N, 1] - Volume densities (non-negative)
        """
        # Encode inputs
        pos_enc = self.pos_encoder.encode(pos)
        dir_enc = self.dir_encoder.encode(view_dir)

        # Process position (first 4 layers)
        x = pos_enc
        for layer in self.pos_layers1:
            x = F.relu(layer(x))

        # Skip connection: concatenate original encoded position
        x = torch.cat([x, pos_enc], dim=-1)

        # Process position (last 4 layers)
        for layer in self.pos_layers2:
            x = F.relu(layer(x))

        # Density (depends only on position)
        sigma = F.relu(self.density_head(x))

        # Feature for color
        feat = self.feature_layer(x)

        # Color (depends on position + direction)
        x = torch.cat([feat, dir_enc], dim=-1)
        for i, layer in enumerate(self.color_layers):
            x = layer(x)
            if i < len(self.color_layers) - 1:
                x = F.relu(x)

        rgb = torch.sigmoid(x)  # RGB in [0, 1]

        return rgb, sigma


# ============================================================================
# Ray Generation and Sampling
# ============================================================================


def get_rays(H, W, focal, c2w):
    """
    Generate rays for all pixels in an image.

    Args:
        H, W: image height and width
        focal: focal length (in pixels)
        c2w: [4,4] camera-to-world matrix (can be on CPU or CUDA)

    Returns:
        rays_o: [H*W, 3] ray origins
        rays_d: [H*W, 3] ray directions (unit vectors)
    """
    # Make sure all tensors are created on the same device as c2w
    if isinstance(c2w, torch.Tensor):
        device = c2w.device
        dtype = c2w.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        c2w = torch.as_tensor(c2w, dtype=dtype, device=device)

    # Pixel coordinates (on same device)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=dtype, device=device),
        torch.arange(H, dtype=dtype, device=device),
        indexing="xy",
    )

    # Camera-space ray directions
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )  # [H, W, 3]

    # Normalize directions
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Rotate ray directions into world space: dirs * R^T
    # c2w[:3, :3] is rotation; broadcasting gives [H, W, 3]
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # [H, W, 3]

    # Ray origins: camera center in world coordinates
    rays_o = c2w[:3, -1].expand_as(rays_d)  # [H, W, 3]

    # Flatten to [N_rays, 3]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    return rays_o, rays_d


def sample_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    """
    Sample points along rays (stratified sampling).

    Args:
        rays_o: [N_rays, 3] Ray origins
        rays_d: [N_rays, 3] Ray directions
        near, far: Near and far bounds
        N_samples: Number of samples per ray
        perturb: Whether to add random jitter within each interval
    Returns:
        pts: [N_rays, N_samples, 3] Sampled 3D points
        z_vals: [N_rays, N_samples] Depth values along the ray
    """
    N_rays = rays_o.shape[0]

    # Linearly spaced samples in [near, far]
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

    # Points along rays: r(t) = o + t * d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


# ============================================================================
# Volume Rendering
# ============================================================================


def volume_render(rgb, sigma, z_vals, rays_d, noise_std=0.0):
    """
    Volume rendering using the classic equation.

    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
    T_i = exp(-sum_{j<i} sigma_j * delta_j)

    Args:
        rgb:   [N_rays, N_samples, 3]  RGB values
        sigma: [N_rays, N_samples, 1]  Densities
        z_vals:[N_rays, N_samples]     Depth values
        rays_d:[N_rays, 3]             Ray directions
        noise_std: Std of noise to add to sigma (regularization)
    Returns:
        rgb_map:  [N_rays, 3] Rendered RGB
        depth_map:[N_rays]   Rendered depth
        acc_map:  [N_rays]   Accumulated opacity
        weights:  [N_rays, N_samples] Weights for each sample
    """
    # Distances between samples along the ray
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)],
        dim=-1,
    )

    # Account for non-unit ray directions by scaling distances
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Optionally add noise to sigma (regularization)
    if noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * noise_std

    # Alpha values
    alpha = 1.0 - torch.exp(-F.relu(sigma[..., 0]) * dists)

    # Transmittance T_i (cumulative product of (1 - alpha))
    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]

    # Weights for each sample
    weights = alpha * transmittance  # [N_rays, N_samples]

    # Render RGB
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # Render depth
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map, weights


# ============================================================================
# Analytic 3D Scene: Colored Sphere
# ============================================================================


def render_sphere_rays(rays_o, rays_d, radius=1.0, center=None, bg_color=None):
    """
    Analytic ground-truth renderer for a simple 3D scene:
    a colored sphere at the origin.

    For each ray, we compute the intersection with the sphere.
    If there is a hit, color = function of surface normal;
    otherwise, background color.

    Args:
        rays_o: [N_rays, 3] ray origins
        rays_d: [N_rays, 3] ray directions
        radius: sphere radius
        center: [3] sphere center (default = [0,0,0])
        bg_color: [3] background RGB (default = [0,0,0])
    Returns:
        colors: [N_rays, 3] RGB in [0,1]
    """
    device = rays_o.device
    if center is None:
        center = torch.zeros(3, device=device)
    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Normalize directions for intersection math
    d = F.normalize(rays_d, dim=-1)
    o = rays_o

    # Ray-sphere intersection: ||o + t d - c||^2 = R^2
    oc = o - center
    a = torch.sum(d * d, dim=-1)  # should be ~1
    b = 2.0 * torch.sum(oc * d, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - radius**2

    discriminant = b * b - 4 * a * c
    hit = discriminant > 0

    # Initialize colors as background
    colors = bg_color.expand(rays_o.shape)

    # For rays that hit the sphere, compute nearest intersection
    if hit.any():
        sqrt_disc = torch.sqrt(discriminant[hit])
        a_hit = a[hit]
        b_hit = b[hit]

        t0 = (-b_hit - sqrt_disc) / (2 * a_hit)
        t1 = (-b_hit + sqrt_disc) / (2 * a_hit)

        # Choose the nearest positive t
        t = torch.where(t0 > 0, t0, t1)
        t = torch.clamp(t, min=0.0)

        # Intersection points
        o_hit = o[hit]
        d_hit = d[hit]
        pts = o_hit + t.unsqueeze(-1) * d_hit  # [N_hit, 3]

        # Surface normals
        normals = F.normalize(pts - center, dim=-1)

        # Simple normal-based coloring (mapped to [0,1])
        col = 0.5 * (normals + 1.0)  # [-1,1] -> [0,1]

        colors[hit] = col

    return colors


# ============================================================================
# Synthetic "NeRF" Dataset using Analytic Scene
# ============================================================================


class SyntheticSphereScene:
    """
    Synthetic scene for NeRF training: a colored sphere at the origin,
    observed from cameras on a circular trajectory.

    We do NOT pre-store images; instead, for each batch:
    - Choose a random camera pose
    - Generate rays
    - Compute ground-truth pixel colors via analytic sphere rendering
    """

    def __init__(
        self, num_images=40, H=64, W=64, focal=64.0, radius=1.0, cam_radius=4.0
    ):
        self.num_images = num_images
        self.H = H
        self.W = W
        self.focal = focal
        self.radius = radius
        self.cam_radius = cam_radius

        # Precompute camera poses on a circle around the origin
        self.poses = self._generate_poses()

    def _generate_poses(self):
        poses = []
        for i in range(self.num_images):
            theta = 2 * np.pi * i / self.num_images
            # Simple orbit around the origin, looking at [0,0,0]
            # Up vector ~ [0,1,0]
            c2w = torch.tensor(
                [
                    [np.cos(theta), 0, np.sin(theta), self.cam_radius * np.sin(theta)],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), self.cam_radius * np.cos(theta)],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
            poses.append(c2w)
        return poses

    def get_batch(self, batch_size=1024, device="cpu"):
        """
        Get a random batch of rays + ground-truth RGB.

        Returns:
            rays_o: [B, 3]
            rays_d: [B, 3]
            target_rgb: [B, 3]
        """
        img_idx = np.random.randint(0, self.num_images)
        pose = self.poses[img_idx].to(device)

        # Generate all rays for this image
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # Ground-truth colors via analytic sphere rendering
        with torch.no_grad():
            full_target = render_sphere_rays(rays_o, rays_d, radius=self.radius)

        # Randomly sample a subset of rays
        indices = torch.randint(0, rays_o.shape[0], (batch_size,), device=device)
        rays_o = rays_o[indices]
        rays_d = rays_d[indices]
        target_rgb = full_target[indices]

        return rays_o, rays_d, target_rgb


# ============================================================================
# Training Function
# ============================================================================


def train_nerf(args):
    """Train NeRF model on the synthetic sphere scene."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Create synthetic scene (analytic 3D sphere)
    # ------------------------------------------------------------------
    print("\nCreating synthetic sphere scene...")
    scene = SyntheticSphereScene(
        num_images=args.num_images,
        H=args.img_size,
        W=args.img_size,
        focal=float(args.img_size),
        radius=1.0,
        cam_radius=4.0,
    )
    print(f"Number of training views: {args.num_images}")
    print(f"Image resolution: {args.img_size}x{args.img_size}")

    # ------------------------------------------------------------------
    # 2. Create NeRF model
    # ------------------------------------------------------------------
    print("\nInitializing NeRF...")
    model = NeRF(pos_L=args.pos_L, dir_L=args.dir_L, hidden_dim=args.hidden_dim).to(
        device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params/1e6:.2f}M")

    # Optimizer + LR scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    print("\nTraining NeRF...")
    model.train()

    pbar = tqdm(range(args.n_iters))
    for it in pbar:
        # (a) Sample random rays + ground-truth pixel colors
        rays_o, rays_d, target_rgb = scene.get_batch(args.batch_size, device=device)

        # (b) Sample points along each ray
        pts, z_vals = sample_along_rays(
            rays_o,
            rays_d,
            near=args.near,
            far=args.far,
            N_samples=args.n_samples,
            perturb=True,
        )

        # (c) Flatten points and directions for MLP: (x, d) -> (c, sigma)
        N_rays, N_samples, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand(N_rays, N_samples, 3).reshape(-1, 3)

        rgb, sigma = model(pts_flat, dirs_flat)

        # (d) Reshape back to [N_rays, N_samples, *]
        rgb = rgb.reshape(N_rays, N_samples, 3)
        sigma = sigma.reshape(N_rays, N_samples, 1)

        # (e) Volume render along each ray to get predicted pixel colors
        rgb_map, depth_map, acc_map, weights = volume_render(
            rgb, sigma, z_vals, rays_d, noise_std=1.0 if it < args.n_iters // 2 else 0.0
        )

        # (f) Photometric loss: predicted vs ground truth RGB
        loss = F.mse_loss(rgb_map, target_rgb)

        # Backprop + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute PSNR for logging
        with torch.no_grad():
            psnr = -10.0 * torch.log10(loss)

        if it % 100 == 0:
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
        "nerf_sphere_model.pth",
    )
    print("Model saved to nerf_sphere_model.pth")

    return model, scene


# ============================================================================
# Rendering Function (Novel Views)
# ============================================================================


@torch.no_grad()
def render_image(
    model, pose, H, W, focal, near, far, N_samples, chunk=1024, device="cuda"
):
    """Render a full image from a camera pose using the trained NeRF."""
    model.eval()

    # Generate rays
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    all_rgb = []
    all_depth = []

    # Render in chunks to avoid OOM
    for i in range(0, rays_o.shape[0], chunk):
        rays_o_chunk = rays_o[i : i + chunk]
        rays_d_chunk = rays_d[i : i + chunk]

        pts, z_vals = sample_along_rays(
            rays_o_chunk, rays_d_chunk, near, far, N_samples, perturb=False
        )

        N_rays, N_samples, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d_chunk[:, None, :].expand(N_rays, N_samples, 3).reshape(-1, 3)

        rgb, sigma = model(pts_flat, dirs_flat)
        rgb = rgb.reshape(N_rays, N_samples, 3)
        sigma = sigma.reshape(N_rays, N_samples, 1)

        rgb_map, depth_map, _, _ = volume_render(
            rgb, sigma, z_vals, rays_d_chunk, noise_std=0.0
        )

        all_rgb.append(rgb_map.cpu())
        all_depth.append(depth_map.cpu())

    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    depth = torch.cat(all_depth, dim=0).reshape(H, W)

    return rgb.numpy(), depth.numpy()


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("NeRF Demo for Novel View Synthesis (Colored Sphere)")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    # Train NeRF on synthetic sphere scene
    model, scene = train_nerf(args)

    # Render novel views
    print("\nRendering novel views...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose a few novel camera poses (offset from training poses)
    test_poses = []
    for i in range(5):
        theta = 2 * np.pi * (i / 5 + 0.1)  # offset angle
        pose = torch.tensor(
            [
                [np.cos(theta), 0, np.sin(theta), scene.cam_radius * np.sin(theta)],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), scene.cam_radius * np.cos(theta)],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        test_poses.append(pose)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, pose in enumerate(test_poses):
        rgb, depth = render_image(
            model,
            pose,
            scene.H,
            scene.W,
            scene.focal,
            args.near,
            args.far,
            args.n_samples,
            device=device,
        )

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"Novel View {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(depth, cmap="viridis")
        axes[1, i].set_title(f"Depth {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("nerf_sphere_results.png", dpi=150, bbox_inches="tight")
    print("Results saved to nerf_sphere_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRF Demo (Synthetic Sphere)")

    # Scene / data
    parser.add_argument(
        "--num_images", type=int, default=40, help="Number of training camera views"
    )
    parser.add_argument("--img_size", type=int, default=64, help="Image size (H=W)")

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
        "--hidden_dim", type=int, default=256, help="Hidden layer width"
    )

    # Training
    parser.add_argument(
        "--n_iters", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (number of rays)"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Rendering / sampling
    parser.add_argument("--near", type=float, default=2.0, help="Near clipping plane")
    parser.add_argument("--far", type=float, default=6.0, help="Far clipping plane")
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of samples per ray"
    )

    args = parser.parse_args()
    main(args)

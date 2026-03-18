import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from textured_gaussians.strategy.ops import rescale_texture


# === Synthetic texture generators ===
def make_checkerboard(width, height):
    y = torch.arange(height).view(-1, 1).expand(height, width)
    x = torch.arange(width).view(1, -1).expand(height, width)
    checker = ((x + y) % 2).float()
    r = checker
    g = 1 - checker
    b = ((x % 3) == 0).float()  # Add a hint of blue for diversity
    img = torch.stack([r, g, b, torch.ones_like(r)])
    return img

def make_horizontal_stripes(width, height, stripe_height=4):
    pattern = torch.arange(height).view(-1, 1) // stripe_height % 2
    stripe = pattern.expand(height, width).float()
    r = stripe
    g = 1 - stripe
    b = (torch.arange(height).view(-1, 1) % 5 == 0).float().expand(height, width)
    img = torch.stack([r, g, b, torch.ones_like(r)])
    return img

def make_vertical_stripes(width, height, stripe_width=4):
    pattern = torch.arange(width).view(1, -1) // stripe_width % 2
    stripe = pattern.expand(height, width).float()
    r = stripe
    g = 0.5 * (1 - stripe)
    b = ((torch.arange(width) % 3) == 0).float().view(1, -1).expand(height, width)
    img = torch.stack([r, g, b, torch.ones_like(r)])
    return img

def make_radial_gradient(width, height):
    y = torch.linspace(-1, 1, height).view(-1, 1).expand(height, width)
    x = torch.linspace(-1, 1, width).view(1, -1).expand(height, width)
    radius = torch.sqrt(x**2 + y**2).clamp(0, 1)
    r = 1 - radius
    g = radius
    b = ((x > 0).float() + (y > 0).float()) / 2
    img = torch.stack([r, g, b, torch.ones_like(r)])
    return img

def make_sine_wave_texture(width, height, freq_x=8, freq_y=8):
    y = torch.linspace(0, 2 * torch.pi * freq_y, height).view(-1, 1)
    x = torch.linspace(0, 2 * torch.pi * freq_x, width).view(1, -1)
    r = torch.sin(x) * 0.5 + 0.5
    g = torch.sin(y) * 0.5 + 0.5
    b = torch.cos(x + y) * 0.5 + 0.5
    img = torch.stack([r.expand_as(b), g.expand_as(b), b, torch.ones_like(b)])
    return img



# === Visualizer ===

def visualize_texture_comparison(
    textures_src: torch.Tensor,
    dims_src: torch.Tensor,
    offsets_src: torch.Tensor,
    textures_dst: torch.Tensor,
    dims_dst: torch.Tensor,
    offsets_dst: torch.Tensor,
    title="Texture Comparison"
):
    n = dims_src.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        Hs, Ws = dims_src[i]
        Hd, Wd = dims_dst[i]
        offs_src = offsets_src[i]
        offs_dst = offsets_dst[i]

        tex_src = textures_src[:, offs_src:offs_src + Hs * Ws].reshape(4, Hs, Ws)
        tex_dst = textures_dst[:, offs_dst:offs_dst + Hd * Wd].reshape(4, Hd, Wd)

        img_src = tex_src[:3].permute(1, 2, 0).numpy()
        img_dst = tex_dst[:3].permute(1, 2, 0).numpy()

        axes[i][0].imshow(img_src)
        axes[i][0].set_title(f"Original #{i} ({Hs}x{Ws})")
        axes[i][0].axis('off')

        axes[i][1].imshow(img_dst)
        axes[i][1].set_title(f"Rescaled #{i} ({Hd}x{Wd})")
        axes[i][1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=200)


# === Main loop ===

def main():
    torch.manual_seed(0)

    dims_src = torch.tensor([
        [1,2],
        [2,1],
        [2,2],
        [2,2],
        [64, 64],
        [48, 32],
        [40, 80]
    ])
    texture_dims_src = dims_src.clone()
    texture_dims_dst = torch.tensor([
        [2,2],
        [2,2],
        [2,4],
        [4,2],
        [32, 32],
        [96, 64],
        [80, 40]
    ])

    pattern_generators = {
        "Checkerboard": make_checkerboard,
        "Horizontal Stripes": make_horizontal_stripes,
        "Vertical Stripes": make_vertical_stripes,
        "Radial Gradient": make_radial_gradient,
        "Sine Waves": make_sine_wave_texture,
    }

    for pattern_name, pattern_fn in pattern_generators.items():
        texture_offsets = torch.zeros((len(dims_src), 1), dtype=torch.long)
        textures = []
        offset = 0
        for i, (H, W) in enumerate(texture_dims_src):
            tex = pattern_fn(W.item(), H.item())
            textures.append(tex.reshape(4, -1))
            texture_offsets[i, 0] = offset
            offset += H.item() * W.item()
        textures_packed = torch.cat(textures, dim=1)

        # Rescale
        textures_rescaled, texture_offsets_dst = rescale_texture(
            textures_packed.clone(),
            texture_offsets.clone(),
            texture_dims_src,
            texture_dims_dst
        )

        # Visualize
        visualize_texture_comparison(
            textures_packed, texture_dims_src, texture_offsets,
            textures_rescaled, texture_dims_dst, texture_offsets_dst,
            title=f"Rescaling Comparison: {pattern_name}"
        )


if __name__ == "__main__":
    main()

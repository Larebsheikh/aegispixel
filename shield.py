"""Image immunization methods for AegisPixel."""
from __future__ import annotations

import numpy as np
from PIL import Image

_VAE = None


def noise_shield(img: Image.Image, epsilon: int, seed: int | None = 42) -> Image.Image:
    """Fast mode: L-infinity bounded sign noise (+/- epsilon per channel, 0-255 scale).

    High-frequency and imperceptible, but NOT optimized against any encoder.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = rng.choice([-1.0, 1.0], size=arr.shape).astype(np.float32) * epsilon
    out = np.clip(arr + noise, 0, 255).round().astype(np.uint8)
    return Image.fromarray(out)


def pgd_available() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _load_vae(device):
    global _VAE
    if _VAE is None:
        from diffusers import AutoencoderKL

        _VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
    return _VAE.to(device)


def pgd_shield(img: Image.Image, epsilon: int, steps: int = 40,
               max_side: int = 512, progress_cb=None) -> Image.Image:
    """PhotoGuard-style encoder attack (Salman et al., 2023).

    Runs Projected Gradient Descent so the Stable Diffusion VAE latent of the
    shielded image is pulled toward the latent of a flat gray target image.
    Image is resized so its longest side <= max_side (multiples of 8).
    """
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _load_vae(device)
    for p in vae.parameters():
        p.requires_grad_(False)

    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    w, h = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)
    img = img.resize((w, h), Image.LANCZOS)

    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    eps = epsilon / 127.5            # epsilon/255 expressed on the [-1, 1] scale
    alpha = 2.5 * eps / steps
    with torch.no_grad():
        target = vae.encode(torch.zeros_like(x)).latent_dist.mean

    delta = (torch.rand_like(x) * 2 - 1) * eps
    for i in range(steps):
        delta.requires_grad_(True)
        adv = torch.clamp(x + delta, -1, 1)
        loss = F.mse_loss(vae.encode(adv).latent_dist.mean, target)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = torch.clamp(delta - alpha * grad.sign(), -eps, eps)
            delta = torch.clamp(x + delta, -1, 1) - x
        if progress_cb:
            progress_cb((i + 1) / steps, float(loss))

    out = torch.clamp(x + delta.detach(), -1, 1)
    out = ((out[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(out)

"""Quick-check metrics (MATLAB remains authoritative).

Lab error follows the ISTD/ShadowFormer convention: per-pixel sum over L,a,b of |Lab(P) - Lab(G)|,
averaged over all / shadow (mask) / non-shadow pixels per image, then averaged over images.
Pooled variants divide the summed error over all images by the summed pixel count.
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

WHITEPOINTS = ("D65", "D50")


def to_lab(rgb_u8, illuminant="D65"):
    return rgb2lab(np.asarray(rgb_u8, dtype=np.float64) / 255.0, illuminant=illuminant)


def lab_error(pred, gt, mask, illuminant="D65"):
    """Per-image Lab error. Returns mean values (None for empty regions) plus raw sums for pooling."""
    d = np.abs(to_lab(pred, illuminant) - to_lab(gt, illuminant)).sum(axis=-1)
    out = {}
    for region, sel in (("all", np.ones_like(mask, dtype=bool)), ("s", mask), ("ns", ~mask)):
        n = int(sel.sum())
        total = float(d[sel].sum())
        out[f"rmse_{region}"] = total / n if n else None
        out[f"sum_{region}"] = total
        out[f"n_{region}"] = n
    return out


def mse(pred, gt):
    return float(((np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)) ** 2).mean())


def psnr(pred, gt):
    m = mse(pred, gt)
    return float("inf") if m == 0 else float(10 * np.log10(255.0**2 / m))


def ssim(pred, gt):
    return float(structural_similarity(np.asarray(pred), np.asarray(gt), channel_axis=-1, data_range=255))


def mae(pred, gt):
    return float(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)).mean())


class LpipsScorer:
    """LPIPS with the ../measure_lpips.py conventions: inputs in [-1, 1], NCHW float32."""

    def __init__(self, nets=("alex", "vgg"), device="cuda"):
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.models = {n: lpips.LPIPS(net=n, verbose=False).to(device).eval() for n in nets}

    def _tensor(self, rgb_u8):
        arr = np.asarray(rgb_u8, dtype=np.float32) / 127.5 - 1.0
        return self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def __call__(self, pred, gt):
        with self.torch.no_grad():
            p, g = self._tensor(pred), self._tensor(gt)
            return {f"lpips_{n}": float(m(p, g).item()) for n, m in self.models.items()}

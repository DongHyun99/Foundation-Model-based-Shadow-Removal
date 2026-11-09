"""Atomic PNG writes and mask compositing."""

import os
from pathlib import Path

import numpy as np
from PIL import Image


def atomic_save_png(img, path):
    """Write a PNG via a dot-prefixed temp file in the same directory, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        img.save(tmp, format="PNG")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def composite(generated, original, mask_l):
    """mask * generated + (1 - mask) * original for a binary {0,255} 'L' mask; pixels outside stay byte-identical."""
    if generated.size != original.size or mask_l.size != original.size:
        raise ValueError(f"size mismatch: generated {generated.size}, original {original.size}, mask {mask_l.size}")
    m = (np.asarray(mask_l) > 0)[..., None]
    out = np.where(m, np.asarray(generated.convert("RGB")), np.asarray(original.convert("RGB")))
    return Image.fromarray(out.astype(np.uint8), mode="RGB")

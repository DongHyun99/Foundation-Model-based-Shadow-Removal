"""Atomic PNG writes, model-size resampling and mask compositing."""

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


RESAMPLE = {"nearest": Image.NEAREST, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def to_model_size(image, mask_l, resolution, io_resample):
    """Image and binary mask at the inference `resolution` (height, width); unchanged if already that size."""
    size = (resolution[1], resolution[0])
    if image.size == size:
        return image, mask_l
    if io_resample is None:
        raise ValueError(f"input size {image.size} != sampling resolution {resolution} and no io_resample configured")
    return (
        image.resize(size, RESAMPLE[io_resample["image_up"]]),
        mask_l.resize(size, RESAMPLE[io_resample["mask_up"]]),
    )


def to_native_size(raw, size, io_resample):
    """Raw model output resized back to the dataset image `size` (width, height)."""
    if raw.size == size:
        return raw
    return raw.resize(size, RESAMPLE[io_resample["output_down"]])


def composite(generated, original, mask_l):
    """mask * generated + (1 - mask) * original for a binary {0,255} 'L' mask; pixels outside stay byte-identical."""
    if generated.size != original.size or mask_l.size != original.size:
        raise ValueError(f"size mismatch: generated {generated.size}, original {original.size}, mask {mask_l.size}")
    m = (np.asarray(mask_l) > 0)[..., None]
    out = np.where(m, np.asarray(generated.convert("RGB")), np.asarray(original.convert("RGB")))
    return Image.fromarray(out.astype(np.uint8), mode="RGB")

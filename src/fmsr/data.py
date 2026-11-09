"""Dataset listing, filename conventions and image/mask loading."""

import os

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_key(name):
    """Strip all trailing image extensions ('_MG_5580.JPG.jpg' -> '_MG_5580'), as ../measure_lpips.py does."""
    stem, ext = os.path.splitext(name)
    while ext.lower() in IMG_EXTS:
        name = stem
        stem, ext = os.path.splitext(name)
    return name


def output_name(name):
    """Output filename: the test_A stem with its last extension replaced by .png."""
    return os.path.splitext(name)[0] + ".png"


def list_images(directory):
    """Sorted image filenames in a directory; dotfiles (e.g. .MATLABDriveTag) and non-images are skipped."""
    return sorted(
        f
        for f in os.listdir(directory)
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def index_by_key(directory):
    """{image_key: filename} for every image in a directory."""
    out = {}
    for f in list_images(directory):
        key = image_key(f)
        if key in out:
            raise ValueError(f"duplicate image key {key!r} in {directory}: {out[key]} / {f}")
        out[key] = f
    return out


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def load_mask_bin(path, threshold):
    """Binary mask as an 'L' image with values {0, 255}: gray > threshold is shadow (white = edit region)."""
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"))
    return Image.fromarray(np.where(gray > threshold, 255, 0).astype(np.uint8), mode="L")

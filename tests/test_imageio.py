import numpy as np
from PIL import Image

from fmsr.imageio import atomic_save_png, composite


def _rand(seed, shape=(8, 8, 3)):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8))


def test_composite_keeps_outside_pixels_exact():
    gen, orig = _rand(0), _rand(1)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    out = np.asarray(composite(gen, orig, Image.fromarray(mask, mode="L")))
    inside = mask > 0
    assert np.array_equal(out[~inside], np.asarray(orig)[~inside])
    assert np.array_equal(out[inside], np.asarray(gen)[inside])


def test_atomic_save_leaves_no_tmp(tmp_path):
    img = _rand(2)
    path = tmp_path / "sub" / "x.png"
    atomic_save_png(img, path)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(img))
    assert [p.name for p in path.parent.iterdir()] == ["x.png"]

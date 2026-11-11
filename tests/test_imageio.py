import numpy as np
from PIL import Image

import pytest

from fmsr.imageio import atomic_save_png, composite, to_model_size, to_native_size

IO = {"image_up": "bicubic", "mask_up": "nearest", "output_down": "bicubic"}


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


def test_model_size_roundtrip_keeps_mask_binary_and_native_size():
    img = _rand(3, (16, 16, 3))
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:9, 2:11] = 255
    m_img, m_mask = to_model_size(img, Image.fromarray(mask, mode="L"), (32, 32), IO)
    assert m_img.size == m_mask.size == (32, 32)
    up = np.asarray(m_mask)
    assert set(np.unique(up)) <= {0, 255}
    assert np.array_equal(up[::2, ::2], mask)  # exact 2x nearest
    assert to_native_size(m_img, img.size, IO).size == (16, 16)


def test_model_size_passthrough_and_missing_config():
    img, mask = _rand(4), Image.new("L", (8, 8))
    assert to_model_size(img, mask, (8, 8), None) == (img, mask)
    assert to_native_size(img, (8, 8), None) is img
    with pytest.raises(ValueError, match="no io_resample"):
        to_model_size(img, mask, (16, 16), None)


def test_atomic_save_leaves_no_tmp(tmp_path):
    img = _rand(2)
    path = tmp_path / "sub" / "x.png"
    atomic_save_png(img, path)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(img))
    assert [p.name for p in path.parent.iterdir()] == ["x.png"]

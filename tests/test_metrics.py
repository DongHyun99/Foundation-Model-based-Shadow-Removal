import numpy as np
import pytest

from fmsr.metrics import lab_error, mae, psnr, ssim, to_lab
from fmsr.sanity import aggregate_summaries, check_output


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, (16, 16, 3), dtype=np.uint8)


def test_identity_is_perfect():
    a = _img(0)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:8] = True
    e = lab_error(a, a, mask)
    assert e["rmse_all"] == e["rmse_s"] == e["rmse_ns"] == 0
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0)
    assert mae(a, a) == 0


def test_lab_error_region_split_and_hand_value():
    gt = np.full((2, 2, 3), 128, dtype=np.uint8)
    pred = gt.copy()
    pred[0, 0] = [255, 255, 255]
    mask = np.array([[True, False], [False, False]])
    d = np.abs(to_lab(pred[0, 0][None, None]) - to_lab(gt[0, 0][None, None])).sum()
    e = lab_error(pred, gt, mask)
    assert e["rmse_s"] == pytest.approx(d)
    assert e["rmse_ns"] == 0
    assert e["rmse_all"] == pytest.approx(d / 4)
    assert (e["n_s"], e["n_ns"], e["n_all"]) == (1, 3, 4)


def test_empty_region_is_none():
    a, b = _img(1), _img(2)
    e = lab_error(a, b, np.zeros((16, 16), dtype=bool))
    assert e["rmse_s"] is None and e["n_s"] == 0
    assert e["rmse_ns"] == pytest.approx(e["rmse_all"])


def test_psnr_mae_known_values():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 10, dtype=np.uint8)
    assert mae(a, b) == 10
    assert psnr(a, b) == pytest.approx(10 * np.log10(255**2 / 100))


def test_aggregate_summaries_by_target_variant():
    def s(v, gen, n, fail, std, ns):
        return {"target_variant": v, "runs": {"i1/default": {"generated": gen}},
                "sanity": {"n": n, "n_fail": fail, "std_min": std, "ns_psnr_min": ns}}

    agg = aggregate_summaries([s("A", 10, 10, 2, 5.0, 20.0), s("A", 30, 30, 1, 1.0, 30.0), s("B", 0, 0, 0, None, None)])
    assert agg["A"] == {"generated": 40, "sanity_n": 40, "sanity_fail": 3, "std_min": 1.0, "ns_psnr_min": 20.0,
                        "sanity_fail_rate": 3 / 40}
    assert agg["B"]["sanity_fail_rate"] is None and agg["B"]["std_min"] is None


def test_sanity_rejects_flat_output():
    from PIL import Image

    orig = Image.fromarray(_img(3))
    mask = Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L")
    assert check_output(orig, orig, mask)["ok"]
    flat = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    assert not check_output(flat, orig, mask)["ok"]

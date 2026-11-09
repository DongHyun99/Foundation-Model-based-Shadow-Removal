"""Numeric output sanity checks (catch black/flat/garbage outputs from broken quantization or pipelines)."""

import numpy as np

STD_MIN = 2.0
NS_PSNR_MIN_DB = 15.0


def check_output(raw, original, mask_l):
    """raw vs input: finite pixels, non-flat image, and the non-shadow region roughly preserved."""
    r = np.asarray(raw.convert("RGB"), dtype=np.float64)
    o = np.asarray(original.convert("RGB"), dtype=np.float64)
    ns = ~(np.asarray(mask_l) > 0)
    finite = bool(np.isfinite(r).all())
    std = float(r.std())
    if ns.any():
        mse = float(((r[ns] - o[ns]) ** 2).mean())
        ns_psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
    else:
        ns_psnr = float("inf")
    return {
        "finite": finite,
        "std": std,
        "ns_psnr": ns_psnr,
        "ok": bool(finite and std > STD_MIN and ns_psnr >= NS_PSNR_MIN_DB),
    }

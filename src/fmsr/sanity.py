"""Numeric output sanity checks (catch black/flat/garbage outputs from broken quantization or pipelines)."""

import numpy as np

STD_MIN = 2.0
NS_PSNR_MIN_DB = 15.0


def aggregate_summaries(summaries):
    """Per target variant: generated count and sanity totals over run_summary dicts.

    Sanity is a diagnostic on the raw output (finite, non-flat, non-shadow region close to the input);
    a failure is a model-output observation, not a pipeline error, so it is reported rather than gated.
    """
    out = {}
    for s in summaries:
        v = out.setdefault(s["target_variant"], {"generated": 0, "sanity_n": 0, "sanity_fail": 0, "std_min": None, "ns_psnr_min": None})
        v["generated"] += sum(r["generated"] for r in s["runs"].values())
        v["sanity_n"] += s["sanity"]["n"]
        v["sanity_fail"] += s["sanity"]["n_fail"]
        for k in ("std_min", "ns_psnr_min"):
            x = s["sanity"][k]
            if x is not None:
                v[k] = x if v[k] is None else min(v[k], x)
    for v in out.values():
        v["sanity_fail_rate"] = v["sanity_fail"] / v["sanity_n"] if v["sanity_n"] else None
    return out


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

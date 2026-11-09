"""Python quick-check metrics (MATLAB evaluation by the user remains authoritative).

For every (variant, prompt, sampling, test, output_type in {raw, composite}) found under OUT_ROOT, plus an
`Input` baseline (test_A vs test_C), compute per image: Lab error all/S/NS (ISTD/ShadowFormer convention,
reported as "rmse"), PSNR, SSIM, MAE on RGB, LPIPS (alex, vgg; inputs in [-1, 1]).

Outputs:
  results/per_image/<Variant>.csv, results/per_image/Input.csv
  results/metrics_summary.csv   (dataset means; *_pooled = total Lab error / total pixels)
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

from fmsr.config import load_config
from fmsr.data import index_by_key, load_rgb
from fmsr.metrics import WHITEPOINTS, LpipsScorer, lab_error, mae, psnr, ssim

OUTPUT_TYPES = ("raw", "composite")
PER_IMAGE_FIELDS = [
    "variant", "prompt_id", "sampling_id", "test", "output_type", "image",
    "rmse_all", "rmse_s", "rmse_ns", "n_s", "n_ns", "psnr", "ssim", "mae", "lpips_alex", "lpips_vgg",
]
SUMMARY_FIELDS = [
    "variant", "prompt_id", "sampling_id", "test", "output_type", "n", "n_missing", "n_psnr_inf",
    "rmse_all", "rmse_s", "rmse_ns", "rmse_all_pooled", "rmse_s_pooled", "rmse_ns_pooled",
    "psnr", "ssim", "mae", "lpips_alex", "lpips_vgg", "lab_whitepoint", "mask_rule",
]


def mask_from(path, rule, threshold):
    from PIL import Image

    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"))
    return gray > (threshold if rule == "gt127" else 0)


def discover(out_root, cfg):
    """(variant, prompt, sampling, output_type, test, dir) for existing output dirs; '_'/'.' dirs skipped."""
    found = []
    for variant in cfg.models.variants:
        vdir = os.path.join(out_root, variant)
        if not os.path.isdir(vdir):
            continue
        for pid in sorted(os.listdir(vdir)):
            pdir = os.path.join(vdir, pid)
            if pid.startswith(("_", ".")) or not os.path.isdir(pdir):
                continue
            for sid in sorted(os.listdir(pdir)):
                if sid.startswith(("_", ".")) or not os.path.isdir(os.path.join(pdir, sid)):
                    continue
                for otype in OUTPUT_TYPES:
                    for test in cfg.datasets.tests:
                        found.append((variant, pid, sid, otype, test, os.path.join(pdir, sid, otype, test)))
    return found


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.6g}"
    return v


def summarize(key, rows, n_expected, whitepoint, mask_rule):
    variant, pid, sid, test, otype = key
    out = dict(variant=variant, prompt_id=pid, sampling_id=sid, test=test, output_type=otype,
               n=len(rows), n_missing=n_expected - len(rows), lab_whitepoint=whitepoint, mask_rule=mask_rule)
    for region in ("all", "s", "ns"):
        vals = [r[f"rmse_{region}"] for r in rows if r[f"rmse_{region}"] is not None]
        out[f"rmse_{region}"] = float(np.mean(vals)) if vals else None
        npx = sum(r[f"n_{region}"] for r in rows)
        out[f"rmse_{region}_pooled"] = sum(r[f"sum_{region}"] for r in rows) / npx if npx else None
    finite = [r["psnr"] for r in rows if not math.isinf(r["psnr"])]
    out["n_psnr_inf"] = len(rows) - len(finite)
    out["psnr"] = float(np.mean(finite)) if finite else None
    for k in ("ssim", "mae", "lpips_alex", "lpips_vgg"):
        out[k] = float(np.mean([r[k] for r in rows])) if rows else None
    return out


def score(pred, gt, mask, lp, whitepoint):
    if pred.shape != gt.shape:
        raise ValueError(f"size mismatch pred {pred.shape} vs gt {gt.shape}")
    r = lab_error(pred, gt, mask, whitepoint)
    r.update(psnr=psnr(pred, gt), ssim=ssim(pred, gt), mae=mae(pred, gt), **lp(pred, gt))
    return r


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k)) for k in fields})
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--lab-whitepoint", choices=WHITEPOINTS, default="D65")
    p.add_argument("--mask-rule", choices=("gt127", "nonzero"), default="gt127")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config()
    ds = cfg.datasets
    lp = LpipsScorer(device=args.device)

    gts, masks, inputs = {}, {}, {}
    for test in ds.tests:
        gts[test] = index_by_key(ds.dir(test, "gt"))
        masks[test] = index_by_key(ds.dir(test, "mask"))
        inputs[test] = index_by_key(ds.dir(test, "shadow"))
        if not (gts[test].keys() == masks[test].keys() == inputs[test].keys()):
            raise ValueError(f"{test}: test_A/B/C keys differ")

    cache = {}

    def gt_mask(test, key):
        if (test, key) not in cache:
            gt = np.asarray(load_rgb(ds.dir(test, "gt") / gts[test][key]))
            m = mask_from(ds.dir(test, "mask") / masks[test][key], args.mask_rule, ds.mask_threshold)
            cache[(test, key)] = (gt, m)
        return cache[(test, key)]

    per_image = defaultdict(list)
    groups = defaultdict(list)
    expected = {}

    # Input baseline
    for test in ds.tests:
        key_ = ("Input", "-", "-", test, "input")
        expected[key_] = len(inputs[test])
        for key, name in inputs[test].items():
            gt, m = gt_mask(test, key)
            r = score(np.asarray(load_rgb(ds.dir(test, "shadow") / name)), gt, m, lp, args.lab_whitepoint)
            groups[key_].append(r)
            per_image["Input"].append(dict(r, variant="Input", prompt_id="-", sampling_id="-", test=test, output_type="input", image=key))

    for variant, pid, sid, otype, test, d in discover(args.out_root, cfg):
        key_ = (variant, pid, sid, test, otype)
        expected[key_] = len(gts[test])
        preds = index_by_key(d) if os.path.isdir(d) else {}
        unknown = preds.keys() - gts[test].keys()
        if unknown:
            raise ValueError(f"{d}: predictions without GT: {sorted(unknown)[:5]}")
        for key, name in sorted(preds.items()):
            gt, m = gt_mask(test, key)
            r = score(np.asarray(load_rgb(os.path.join(d, name))), gt, m, lp, args.lab_whitepoint)
            groups[key_].append(r)
            per_image[variant].append(dict(r, variant=variant, prompt_id=pid, sampling_id=sid, test=test, output_type=otype, image=key))
        print(f"{variant}/{pid}/{sid}/{otype}/{test}: {len(preds)}/{expected[key_]}", file=sys.stderr, flush=True)

    summary = [summarize(k, rows, expected[k], args.lab_whitepoint, args.mask_rule) for k, rows in groups.items()]
    summary += [summarize(k, [], expected[k], args.lab_whitepoint, args.mask_rule) for k in expected if k not in groups]
    summary.sort(key=lambda s: (s["variant"] != "Input", s["variant"], s["prompt_id"], s["sampling_id"], s["test"], s["output_type"]))

    for name, rows in per_image.items():
        write_csv(os.path.join(args.results, "per_image", f"{name}.csv"), PER_IMAGE_FIELDS, rows)
    write_csv(os.path.join(args.results, "metrics_summary.csv"), SUMMARY_FIELDS, summary)
    missing = sum(s["n_missing"] for s in summary)
    print(f"wrote {len(summary)} summary rows; n_missing total = {missing}", file=sys.stderr)
    if args.require_complete and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

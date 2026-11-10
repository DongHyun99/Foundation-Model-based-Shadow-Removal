"""Smoke test before the full run: 4 variants x 3 datasets x 2 images with hard gates.

(a) 4 workers in parallel (GPUs 0, 2, 4, 6), prompts i1/e1, sampling default, OUT_ROOT=outputs/_smoke.
(b) gates per variant: 6 tasks done, outputs at the sampling resolution, composite byte-identical to the
    input outside the mask, conditioning preprocess size == resolution, NVML peak <= vram_limit_mib,
    output sanity (finite, std > 2, raw-vs-input non-shadow PSNR >= 15 dB).
(c) resume: rerunning every worker exits 3 without loading and leaves outputs untouched.
(d) steal drill: only GPU0 (Fill) is mapped; it must drain its residual i1 work, then steal
    Qwen-Image-Inpaint i1 work (no owner) via the supervisor, without touching existing outputs.
Writes reports/smoke_report.json and exits non-zero on any gate failure.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from fmsr import tasks as T
from fmsr.config import REPO_ROOT, load_config
from fmsr.data import load_mask_bin, load_rgb

OUT_ROOT = "outputs/_smoke"
SMOKE_GPUS = {"FLUX.1-Fill-dev": 0, "FLUX.1-Kontext-dev": 2, "Qwen-Image-Inpaint": 4, "Qwen-Image-Edit-2509": 6}
PROMPTS = ["i1", "e1"]
SAMPLING = "default"
LIMIT = 2
STEAL_LIMIT = 4
PY = str(REPO_ROOT / "venv/bin/python")


def env(**extra):
    e = dict(os.environ, OUT_ROOT=OUT_ROOT, PROMPTS=" ".join(PROMPTS), SAMPLINGS=SAMPLING, LIMIT_PER_DATASET=str(LIMIT))
    e.pop("MIN_REMAINING", None)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    e.update(extra)
    return e


def run_workers(log_dir, tag):
    procs = {}
    for variant, gpu in SMOKE_GPUS.items():
        log = open(os.path.join(log_dir, f"{tag}_{variant}.log"), "w")
        procs[variant] = (
            subprocess.Popen(
                [PY, "scripts/run.py", "--variant", variant, "--gpu", str(gpu)],
                env=env(CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            ),
            log,
        )
    codes = {}
    for variant, (p, log) in procs.items():
        codes[variant] = p.wait()
        log.close()
    return codes


def summaries(variant):
    return sorted(glob.glob(os.path.join(OUT_ROOT, variant, "run_summary_*.json")))


def output_mtimes():
    return {p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(OUT_ROOT, "*/*/*/*/*/*.png"))}


def check_variant(cfg, variant, limit, summary_paths, waivers=None):
    tasks = T.enumerate_tasks(cfg, variant, PROMPTS, [SAMPLING], limit)
    res = cfg.samplings[SAMPLING].resolution
    ds = cfg.datasets
    n_ok, size_ok, outside_equal = 0, True, True
    for t in tasks:
        if not T.is_done(OUT_ROOT, t):
            continue
        n_ok += 1
        image = load_rgb(ds.dir(t.test, "shadow") / t.image)
        mask = np.asarray(load_mask_bin(ds.dir(t.test, "mask") / t.image, ds.mask_threshold)) > 0
        raw = Image.open(T.raw_path(OUT_ROOT, t))
        comp = np.asarray(Image.open(T.composite_path(OUT_ROOT, t)).convert("RGB"))
        size_ok &= raw.size == (res[1], res[0]) and comp.shape[:2] == res
        outside_equal &= bool(np.array_equal(comp[~mask], np.asarray(image)[~mask]))
    peaks, pre_sizes, sanity = [], set(), {"n": 0, "n_fail": 0, "std_min": None, "ns_psnr_min": None}
    for path in summary_paths:
        s = json.load(open(path))
        peaks.append(s["vram"]["nvml_peak_mib"])
        pre_sizes |= {tuple(x) for x in s["preprocess_sizes"]}
        for k in ("n", "n_fail"):
            sanity[k] += s["sanity"][k]
        for k in ("std_min", "ns_psnr_min"):
            if s["sanity"][k] is not None:
                sanity[k] = s["sanity"][k] if sanity[k] is None else min(sanity[k], s["sanity"][k])
    last = json.load(open(summary_paths[-1])) if summary_paths else {}
    report = {
        "n_expected": len(tasks),
        "n_ok": n_ok,
        "size_ok": bool(size_ok),
        "outside_equal": bool(outside_equal),
        "cond_sizes": sorted(pre_sizes),
        "vram": last.get("vram"),
        "nvml_peak_mib": max(peaks) if peaks else None,
        "mean_s_per_img": last.get("mean_s_per_img"),
        "load_seconds": last.get("load_seconds"),
        "sanity": sanity,
    }
    waiver = (waivers or {}).get(variant)
    report["sanity_waiver"] = waiver
    report["pass"] = bool(
        n_ok == len(tasks)
        and size_ok
        and outside_equal
        and pre_sizes == {tuple(res)}
        and peaks
        and max(peaks) <= cfg.models.vram_limit_mib
        and sanity["n"] > 0
        and (sanity["n_fail"] == 0 or waiver is not None)
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-steal", action="store_true")
    p.add_argument(
        "--sanity-waiver",
        action="append",
        default=[],
        metavar="VARIANT=REASON",
        help="Accept sanity failures for a variant with documented evidence (recorded in the report)",
    )
    args = p.parse_args()
    waivers = dict(w.split("=", 1) for w in args.sanity_waiver)
    os.chdir(REPO_ROOT)
    cfg = load_config()
    log_dir = os.path.join("logs", "smoke_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(log_dir, exist_ok=True)
    report = {"out_root": OUT_ROOT, "vram_limit_mib": cfg.models.vram_limit_mib, "logs": log_dir, "sanity_waivers": waivers}

    # (a) + (b)
    t0 = time.time()
    report["first_exit_codes"] = run_workers(log_dir, "first")
    report["first_seconds"] = round(time.time() - t0, 1)
    report["variants"] = {v: check_variant(cfg, v, LIMIT, summaries(v), waivers) for v in SMOKE_GPUS}

    # (c) resume
    before = output_mtimes()
    n_summaries = {v: len(summaries(v)) for v in SMOKE_GPUS}
    t0 = time.time()
    codes = run_workers(log_dir, "resume")
    report["resume"] = {
        "exit_codes": codes,
        "seconds": round(time.time() - t0, 1),
        "outputs_unchanged": output_mtimes() == before,
        "n_outputs": len(before),
        "no_new_summaries": all(len(summaries(v)) == n_summaries[v] for v in SMOKE_GPUS),
    }
    report["resume"]["pass"] = (
        all(c == 3 for c in codes.values())
        and report["resume"]["outputs_unchanged"]
        and report["resume"]["no_new_summaries"]
        and len(before) == 2 * sum(r["n_expected"] for r in report["variants"].values())
    )

    # (d) steal drill: only the thief is mapped.
    if not args.skip_steal:
        before = output_mtimes()
        thief, target = "FLUX.1-Fill-dev", "Qwen-Image-Inpaint"
        target_before = set(summaries(target))
        t0 = time.time()
        code = subprocess.call(
            ["bash", "scripts/launch_all.sh"],
            env=env(GPU_MAP=f"{SMOKE_GPUS[thief]}:{thief}", PROMPTS="i1", LIMIT_PER_DATASET=str(STEAL_LIMIT), MIN_REMAINING="1", STAGGER_S="0", FOREGROUND="1"),
            cwd=REPO_ROOT,
        )
        new = sorted(set(summaries(target)) - target_before)
        steal = [json.load(open(s)) for s in new]
        after = output_mtimes()
        drill = {
            "launcher_exit": code,
            "seconds": round(time.time() - t0, 1),
            "steal_summaries": new,
            "is_steal": [s["is_steal"] for s in steal],
            "generated": sum(r["generated"] for s in steal for r in s["runs"].values()),
            "existing_unchanged": all(after.get(k) == v for k, v in before.items()),
            "target": check_variant(cfg, target, STEAL_LIMIT, new),
            "thief": check_variant(cfg, thief, STEAL_LIMIT, sorted(set(summaries(thief)))[-1:], waivers),
        }
        drill["pass"] = bool(
            code == 0
            and steal
            and all(drill["is_steal"])
            and drill["generated"] > 0
            and drill["existing_unchanged"]
            and drill["target"]["pass"]
            and drill["thief"]["pass"]
        )
        report["steal_drill"] = drill

    report["pass"] = (
        all(r["pass"] for r in report["variants"].values())
        and report["resume"]["pass"]
        and report.get("steal_drill", {"pass": True})["pass"]
    )
    os.makedirs("reports", exist_ok=True)
    with open("reports/smoke_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=list)
        f.write("\n")
    print(json.dumps(report, indent=2, default=list))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

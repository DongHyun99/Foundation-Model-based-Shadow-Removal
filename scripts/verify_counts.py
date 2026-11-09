"""Count expected vs present outputs per run, validate run_config.yaml records, list .failed tasks.

Writes reports/counts.json. With --expect-complete, exits 1 unless every run is complete with no failures.
Directories starting with '_' or '.' (outputs/_smoke, outputs/.claims) are not part of the full run.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import yaml

from fmsr import tasks as T
from fmsr.config import load_config

REQUIRED_KEYS = [
    "variant",
    "hf_repo",
    "hf_revision",
    "pipeline_class",
    "quantization",
    "prompt_id",
    "prompt_text",
    "sampling_id",
    "sampling_params",
    "seed",
    "resolution",
    "versions",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--expect-complete", action="store_true")
    p.add_argument("--report", default="reports/counts.json")
    args = p.parse_args()

    cfg = load_config()
    runs, total = {}, {"expected_tasks": 0, "raw": 0, "composite": 0, "missing_tasks": 0, "failed": 0}
    for variant in cfg.models.variants:
        tasks = T.enumerate_tasks(cfg, variant)
        by_run = {}
        for t in tasks:
            by_run.setdefault(str(t.run_rel), []).append(t)
        for rel, ts in by_run.items():
            raw = sum(T.raw_path(args.out_root, t).exists() for t in ts)
            comp = sum(T.composite_path(args.out_root, t).exists() for t in ts)
            missing = sum(not T.is_done(args.out_root, t) for t in ts)
            failed = [str(T.failed_path(args.out_root, t)) for t in ts if T.is_failed(args.out_root, t)]
            rc_path = os.path.join(args.out_root, rel, "run_config.yaml")
            rc_missing_keys = None
            if os.path.exists(rc_path):
                rc = yaml.safe_load(open(rc_path))
                rc_missing_keys = [k for k in REQUIRED_KEYS if rc.get(k) in (None, "", {})]
            runs[rel] = {
                "expected": len(ts),
                "raw": raw,
                "composite": comp,
                "missing_tasks": missing,
                "failed": failed,
                "run_config": os.path.exists(rc_path),
                "run_config_missing_keys": rc_missing_keys,
            }
            total["expected_tasks"] += len(ts)
            total["raw"] += raw
            total["composite"] += comp
            total["missing_tasks"] += missing
            total["failed"] += len(failed)
            for f in failed:
                os.makedirs("logs/failed", exist_ok=True)
                shutil.copy(f, os.path.join("logs/failed", rel.replace("/", "__") + "__" + os.path.basename(f)))

    total["expected_pngs"] = 2 * total["expected_tasks"]
    total["present_pngs"] = total["raw"] + total["composite"]
    total["runs"] = len(runs)
    total["run_configs_ok"] = sum(r["run_config"] and r["run_config_missing_keys"] == [] for r in runs.values())
    # glob skips dot-dirs; also skip '_'-prefixed run dirs (e.g. _smoke). File names may start with '_'.
    on_disk = [
        p
        for p in glob.glob(os.path.join(args.out_root, "*", "*", "*", "*", "*", "*.png"))
        if not any(part.startswith("_") for part in os.path.relpath(os.path.dirname(p), args.out_root).split(os.sep))
    ]
    total["png_files_on_disk"] = len(on_disk)
    complete = (
        total["missing_tasks"] == 0
        and total["failed"] == 0
        and total["run_configs_ok"] == len(runs)
        and total["png_files_on_disk"] == total["expected_pngs"]
    )
    total["complete"] = complete
    report = {"total": total, "runs": runs}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(total, indent=2))
    if args.expect_complete and not complete:
        sys.exit(1)


if __name__ == "__main__":
    main()

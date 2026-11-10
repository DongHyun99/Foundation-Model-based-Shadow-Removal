"""One GPU worker: loads exactly one variant and generates its unfinished, unclaimed tasks.

Exit codes: 3 = no claimable work left for this variant (normal end), other non-zero = crash/OOM.
`--pick-next` mode prints the variant this GPU should work on next (or nothing) without loading a model.

Run with CUDA_VISIBLE_DEVICES=<gpu>; --gpu is the physical index (NVML, summaries, claims).
"""

import argparse
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

import torch

from fmsr import tasks as T
from fmsr.adapters import get_adapter_class
from fmsr.config import load_config
from fmsr.data import load_mask_bin, load_rgb
from fmsr.imageio import atomic_save_png, composite, to_model_size, to_native_size
from fmsr.runinfo import atomic_write_json, build_run_config, write_run_config
from fmsr.sanity import check_output
from fmsr.vram import VramSampler, gpu_info

EXIT_DRAINED = 3
SUMMARY_EVERY = 50


def env_int(name):
    """Integer env override; unset or empty means no override."""
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant")
    p.add_argument("--home", help="Home variant of this GPU (default: --variant); marks stolen work")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out-root", default=os.environ.get("OUT_ROOT") or "outputs")
    p.add_argument("--prompts", nargs="*", default=os.environ.get("PROMPTS", "").split() or None)
    p.add_argument("--samplings", nargs="*", default=os.environ.get("SAMPLINGS", "").split() or None)
    p.add_argument("--limit-per-dataset", type=int, default=env_int("LIMIT_PER_DATASET"))
    p.add_argument("--pick-next", action="store_true")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--variants", nargs="*", default=os.environ.get("VARIANTS", "").split() or None,
                   help="--pick-next candidates (default: every variant)")
    p.add_argument("--min-remaining", type=int, default=env_int("MIN_REMAINING"))
    p.add_argument("--stale-after-s", type=float, default=T.DEFAULT_STALE_AFTER_S)
    args = p.parse_args()
    if not args.pick_next and not args.variant:
        p.error("--variant is required unless --pick-next")
    return args


def variant_tasks(cfg, variant, args):
    return T.enumerate_tasks(cfg, variant, args.prompts, args.samplings, args.limit_per_dataset)


def pick_next(cfg, args):
    remaining, weights, mins = {}, {}, {}
    for name, v in cfg.models.variants.items():
        if args.variants and name not in args.variants:
            continue
        remaining[name] = T.remaining_unclaimed(args.out_root, variant_tasks(cfg, name, args), args.stale_after_s)
        weights[name] = v.cost_weight
        mins[name] = args.min_remaining or v.min_remaining
    target = T.pick_next(remaining, weights, mins, exclude=set(args.exclude))
    print(f"pick-next gpu{args.gpu}: remaining={remaining} -> {target}", file=sys.stderr)
    if target:
        print(target)
    return 0


class Summary:
    def __init__(self, args, variant, home):
        self.start = datetime.now(timezone.utc)
        self.path = os.path.join(
            args.out_root,
            variant,
            f"run_summary_{socket.gethostname()}_gpu{args.gpu}_{self.start:%Y%m%dT%H%M%SZ}.json",
        )
        self.data = {
            "home_variant": home,
            "target_variant": variant,
            "is_steal": variant != home,
            "gpu": args.gpu,
            "pid": os.getpid(),
            "filters": {"prompts": args.prompts, "samplings": args.samplings, "limit_per_dataset": args.limit_per_dataset},
            "start_utc": self.start.isoformat(),
            "end_utc": None,
            "exit_code": None,
            "runs": {},
            "gen_seconds": 0.0,
            "load_seconds": None,
            "preprocess_sizes": [],
            "sanity": {"n": 0, "n_fail": 0, "std_min": None, "ns_psnr_min": None, "failures": []},
        }

    def run(self, task):
        key = f"{task.prompt_id}/{task.sampling_id}"
        return self.data["runs"].setdefault(key, {"generated": 0, "skipped": 0, "failed": 0})

    def add_sanity(self, task, result):
        s = self.data["sanity"]
        s["n"] += 1
        s["std_min"] = result["std"] if s["std_min"] is None else min(s["std_min"], result["std"])
        ns = result["ns_psnr"] if math.isfinite(result["ns_psnr"]) else 1e9
        s["ns_psnr_min"] = ns if s["ns_psnr_min"] is None else min(s["ns_psnr_min"], ns)
        if not result["ok"]:
            s["n_fail"] += 1
            if len(s["failures"]) < 20:
                s["failures"].append({"task": f"{task.run_rel}/{task.test}/{task.image}", **result})

    def write(self, sampler=None, adapter=None, exit_code=None):
        gen = sum(r["generated"] for r in self.data["runs"].values())
        self.data["mean_s_per_img"] = self.data["gen_seconds"] / gen if gen else None
        if sampler is not None:
            self.data["vram"] = sampler.report()
        if adapter is not None:
            self.data["preprocess_sizes"] = sorted({tuple(s) for s in adapter.preprocess_sizes})
        if exit_code is not None:
            self.data["exit_code"] = exit_code
            self.data["end_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.path, self.data)


def work(cfg, args):
    variant = cfg.models.variants[args.variant]
    home = args.home or args.variant
    tasks = variant_tasks(cfg, args.variant, args)
    if T.remaining_unclaimed(args.out_root, tasks, args.stale_after_s) == 0:
        print(f"{args.variant}: nothing left to claim; exiting without loading", file=sys.stderr)
        return EXIT_DRAINED

    # The two GPUs of a pair start at different offsets to minimize claim contention.
    home_gpus = list(variant.gpus)
    offset = (home_gpus.index(args.gpu) * len(tasks) // len(home_gpus)) if args.gpu in home_gpus else 0
    tasks = tasks[offset:] + tasks[:offset]

    summary = Summary(args, args.variant, home)
    sampler = VramSampler(args.gpu).start()
    sampler.mark("load")
    t0 = time.time()
    adapter = get_adapter_class(variant.adapter)(variant, device="cuda", on_phase=sampler.mark)
    adapter.load()
    runs = cfg.runs(args.variant, args.prompts, args.samplings)
    # Always prepare the variant's full prompt set so load/encode memory matches the full run.
    adapter.prepare_prompts(cfg.prompts_for(args.variant))
    summary.data["load_seconds"] = time.time() - t0
    describe = adapter.describe()

    expected = {test: (min(n, args.limit_per_dataset) if args.limit_per_dataset else n) for test, n in cfg.datasets.tests.items()}
    info = gpu_info(args.gpu)
    for run in runs:
        params = run.sampling.params(args.variant)
        rc = build_run_config(
            run=run,
            describe=describe,
            sampling_params=params,
            resize_overrides=adapter.resize_overrides(params, run.sampling.resolution),
            datasets=cfg.datasets,
            expected_counts=expected,
            gpu_info=info,
        )
        write_run_config(T.run_dir(args.out_root, args.variant, run.prompt_id, run.sampling.id) / "run_config.yaml", rc)
    run_by_key = {(r.prompt_id, r.sampling.id): r for r in runs}

    ds = cfg.datasets
    n_done = 0
    for task in tasks:
        stats = summary.run(task)
        if T.is_done(args.out_root, task):
            stats["skipped"] += 1
            continue
        if not T.try_claim(args.out_root, task, args.gpu, args.stale_after_s):
            continue
        run = run_by_key[(task.prompt_id, task.sampling_id)]
        try:
            image = load_rgb(ds.dir(task.test, "shadow") / task.image)
            mask = load_mask_bin(ds.dir(task.test, "mask") / task.image, ds.mask_threshold)
            model_image, model_mask = to_model_size(image, mask, run.sampling.resolution, run.sampling.io_resample)
            generator = torch.Generator(run.sampling.generator_device).manual_seed(run.sampling.seed)
            t1 = time.time()
            raw = adapter.generate(model_image, model_mask, task.prompt_id, run.prompt_text, run.sampling.params(args.variant), run.sampling.resolution, generator)
            summary.data["gen_seconds"] += time.time() - t1
            raw = to_native_size(raw, image.size, run.sampling.io_resample)
            atomic_save_png(raw, T.raw_path(args.out_root, task))
            atomic_save_png(composite(raw, image, mask), T.composite_path(args.out_root, task))
            summary.add_sanity(task, check_output(raw, image, mask))
            stats["generated"] += 1
            T.release(args.out_root, task)
        except torch.cuda.OutOfMemoryError:
            T.release(args.out_root, task)
            summary.write(sampler, adapter, exit_code=1)
            raise
        except Exception:
            err = traceback.format_exc()
            print(f"FAILED {task}: {err}", file=sys.stderr)
            T.mark_failed(args.out_root, task, err, args.gpu)
            stats["failed"] += 1
        n_done += 1
        if n_done % SUMMARY_EVERY == 0:
            summary.write(sampler, adapter)
            print(f"{args.variant} gpu{args.gpu}: {n_done} generated, {summary.data['mean_s_per_img']:.2f} s/img", file=sys.stderr, flush=True)

    sampler.stop()
    summary.write(sampler, adapter, exit_code=EXIT_DRAINED)
    print(f"{args.variant} gpu{args.gpu}: drained; summary {summary.path}", file=sys.stderr)
    return EXIT_DRAINED


def main():
    args = parse_args()
    cfg = load_config()
    if args.pick_next:
        return pick_next(cfg, args)
    return work(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

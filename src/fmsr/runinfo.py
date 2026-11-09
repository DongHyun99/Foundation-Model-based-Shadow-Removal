"""Per-run config records (no-clobber, whitelisted comparison) and atomic per-process summaries."""

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from fmsr.config import REPO_ROOT

VERSION_PACKAGES = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "numpy",
    "pillow",
]

# Only these keys must match when a run_config.yaml already exists. Machine/time/git fields are informational.
REPRO_KEYS = [
    "variant",
    "hf_repo",
    "hf_revision",
    "pipeline_class",
    "adapter",
    "quantization",
    "text_encoder_handling",
    "prompt_id",
    "prompt_text",
    "negative_prompt",
    "sampling_id",
    "sampling_params",
    "seed",
    "generator_device",
    "resolution",
    "resize_overrides",
    "mask_threshold",
    "versions.python",
    "versions.torch",
    "versions.torch_cuda",
    "versions.diffusers",
    "versions.transformers",
    "versions.accelerate",
    "versions.bitsandbytes",
    "versions.huggingface_hub",
    "versions.numpy",
    "versions.pillow",
]


class RunConfigMismatch(RuntimeError):
    pass


def package_versions():
    import torch

    out = {"python": platform.python_version(), "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for pkg in VERSION_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {"repo_git_sha": run("rev-parse", "HEAD"), "repo_dirty": bool(run("status", "--porcelain"))}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_config(*, run, describe, sampling_params, resize_overrides, datasets, expected_counts, gpu_info):
    return {
        "variant": run.variant.name,
        "hf_repo": run.variant.hf_repo,
        "hf_revision": describe["hf_revision"],
        "pipeline_class": describe["pipeline_class"],
        "adapter": run.variant.adapter,
        "kind": run.variant.kind,
        "quantization": describe["quantization"],
        "text_encoder_handling": describe["text_encoder_handling"],
        "prompt_id": run.prompt_id,
        "prompt_text": run.prompt_text,
        "negative_prompt": sampling_params.get("negative_prompt"),
        "sampling_id": run.sampling.id,
        "sampling_params": sampling_params,
        "seed": run.sampling.seed,
        "generator_device": run.sampling.generator_device,
        "resolution": list(run.sampling.resolution),
        "resize_overrides": resize_overrides,
        "mask_threshold": datasets.mask_threshold,
        "dataset_root": str(datasets.root),
        "expected_counts": expected_counts,
        "versions": package_versions(),
        "driver_version": gpu_info.get("driver_version"),
        "gpu_name": gpu_info.get("gpu_name"),
        **git_state(),
        "host": socket.gethostname(),
        "created_at": utc_now(),
    }


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _normalize(value):
    # YAML round-trip turns tuples into lists; compare through JSON.
    return json.loads(json.dumps(value, sort_keys=True))


def compare_repro(existing, new):
    """List of REPRO_KEYS whose values differ."""
    return [k for k in REPRO_KEYS if _normalize(_get(existing, k)) != _normalize(_get(new, k))]


def write_run_config(path, cfg):
    """Create run_config.yaml without clobbering; if it exists, require REPRO_KEYS to match."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    try:
        os.link(tmp, path)
        return "created"
    except FileExistsError:
        existing = yaml.safe_load(path.read_text())
        diff = compare_repro(existing, cfg)
        if diff:
            raise RunConfigMismatch(f"{path}: reproducibility keys differ from existing record: {diff}")
        return "matched"
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

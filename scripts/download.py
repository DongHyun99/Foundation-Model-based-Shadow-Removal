"""Download the diffusers-format weights for every variant and record resolved revisions.

Only the diffusers component folders are fetched (root single-file checkpoints are skipped).
Writes reports/download.json with {variant: {hf_repo, revision, local_path}}.
"""

import argparse
import json
import os

from huggingface_hub import HfApi, snapshot_download

from fmsr.config import load_models

ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder*/*",
    "tokenizer*/*",
    "processor/*",
    "transformer/*",
    "vae/*",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="configs/models.yaml")
    p.add_argument("--variants", nargs="*", help="Subset of variant names (default: all)")
    p.add_argument("--out", default="reports/download.json")
    p.add_argument("--max-workers", type=int, default=16)
    args = p.parse_args()

    variants = load_models(args.models).variants
    names = args.variants or list(variants)
    report = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            report = json.load(f)

    api = HfApi()
    for name in names:
        repo = variants[name].hf_repo
        revision = api.model_info(repo).sha
        print(f"downloading {name}: {repo}@{revision}", flush=True)
        path = snapshot_download(repo, revision=revision, allow_patterns=ALLOW_PATTERNS, max_workers=args.max_workers)
        report[name] = {"hf_repo": repo, "revision": revision, "local_path": path}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

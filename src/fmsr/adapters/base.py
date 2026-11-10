"""Adapter interface: one variant = one pipeline; the runner never touches diffusers directly."""

import json
from abc import ABC, abstractmethod

import torch

from fmsr.config import REPO_ROOT

DOWNLOAD_REPORT = REPO_ROOT / "reports" / "download.json"


def pinned_revision(hf_repo):
    """HF snapshot revision recorded by scripts/download.py; variants sharing a repo share the revision."""
    if not DOWNLOAD_REPORT.exists():
        raise FileNotFoundError(f"{DOWNLOAD_REPORT} missing: run scripts/download.py first")
    revisions = {e["revision"] for e in json.loads(DOWNLOAD_REPORT.read_text()).values() if e["hf_repo"] == hf_repo}
    if len(revisions) != 1:
        raise KeyError(f"{hf_repo}: expected one pinned revision in {DOWNLOAD_REPORT}, found {sorted(revisions)}")
    return revisions.pop()


class Adapter(ABC):
    """Lifecycle: load() -> prepare_prompts({pid: text}) -> generate(...) per image."""

    pipeline_class = None

    def __init__(self, variant, device="cuda", on_phase=None):
        self.variant = variant
        self.device = torch.device(device)
        self.revision = pinned_revision(variant.hf_repo)
        self.on_phase = on_phase or (lambda phase: None)
        self.pipe = None
        self.preprocess_sizes = []

    @abstractmethod
    def load(self):
        """Load weights onto the GPU (or the first stage of a staged load)."""

    @abstractmethod
    def prepare_prompts(self, prompt_map):
        """Called once with every prompt id the process may use."""

    @abstractmethod
    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        """Return the raw model output as a PIL RGB image of exactly `resolution`."""

    @abstractmethod
    def describe(self):
        """Dict with pipeline_class, hf_revision, quantization (as applied, with verified counts), text_encoder_handling."""

    def resize_overrides(self, params, resolution):
        return {}

    def _track_preprocess(self):
        """Record the (height, width) every conditioning image is preprocessed to (smoke asserts the resolution)."""
        processor = self.pipe.image_processor
        original = processor.preprocess

        def preprocess(image, height=None, width=None, *args, **kwargs):
            self.preprocess_sizes.append((height, width))
            return original(image, height, width, *args, **kwargs)

        processor.preprocess = preprocess

    def _check_output(self, image, resolution):
        h, w = resolution
        if image.size != (w, h):
            raise RuntimeError(f"{self.variant.name}: output size {image.size} != requested {(w, h)}")
        return image.convert("RGB")

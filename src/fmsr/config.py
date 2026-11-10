"""YAML configuration loading (models, prompts, sampling, datasets) and the run matrix."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from fmsr.imageio import RESAMPLE

REPO_ROOT = Path(__file__).resolve().parents[2]
KINDS = ("inpaint", "edit", "edit_mask")
CONFIG_DIR = REPO_ROOT / "configs"


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Variant:
    name: str
    hf_repo: str
    adapter: str
    kind: str  # prompt set: "inpaint" | "edit" | "edit_mask" (mask passed as a second image)
    gpus: tuple
    cost_weight: float
    min_remaining: int
    quantization: dict


@dataclass(frozen=True)
class ModelsConfig:
    variants: dict  # name -> Variant (file order preserved)
    vram_limit_mib: int


def load_models(path=CONFIG_DIR / "models.yaml"):
    raw = _read_yaml(path)
    variants = {}
    for name, v in raw["variants"].items():
        if v["kind"] not in KINDS:
            raise ValueError(f"{name}: kind must be one of {KINDS}, got {v['kind']!r}")
        variants[name] = Variant(
            name=name,
            hf_repo=v["hf_repo"],
            adapter=v["adapter"],
            kind=v["kind"],
            gpus=tuple(v["gpus"]),
            cost_weight=float(v["cost_weight"]),
            min_remaining=int(v["min_remaining"]),
            quantization=v["quantization"],
        )
    return ModelsConfig(variants=variants, vram_limit_mib=int(raw["vram_limit_mib"]))


def load_prompts(path=CONFIG_DIR / "prompts.yaml"):
    """Return {kind: {prompt_id: text}}. Prompt IDs must be unique across kinds."""
    raw = _read_yaml(path)
    seen = set()
    out = {}
    for kind in KINDS:
        prompts = {str(k): ("" if v is None else str(v)) for k, v in raw[kind].items()}
        dup = seen & prompts.keys()
        if dup:
            raise ValueError(f"duplicate prompt ids across kinds: {sorted(dup)}")
        seen |= prompts.keys()
        out[kind] = prompts
    return out


@dataclass(frozen=True)
class Sampling:
    id: str
    seed: int
    generator_device: str
    resolution: tuple  # (height, width) of model inference
    per_variant: dict
    io_resample: dict | None = None  # {image_up, mask_up, output_down} when inference size != dataset size

    def params(self, variant_name):
        """Sampling params for one variant with `from_resolution` placeholders resolved."""
        if variant_name not in self.per_variant:
            raise KeyError(f"sampling {self.id!r} has no params for variant {variant_name!r}")
        area = self.resolution[0] * self.resolution[1]
        return {k: (area if v == "from_resolution" else v) for k, v in self.per_variant[variant_name].items()}


def load_sampling(path=CONFIG_DIR / "sampling.yaml"):
    raw = _read_yaml(path)
    out = {}
    for sid, s in raw.items():
        h, w = s["resolution"]
        if h % 16 or w % 16:
            raise ValueError(f"sampling {sid}: resolution must be a multiple of 16, got {h}x{w}")
        io_resample = s.get("io_resample")
        if io_resample is not None:
            if set(io_resample) != {"image_up", "mask_up", "output_down"}:
                raise ValueError(f"sampling {sid}: io_resample needs exactly image_up, mask_up, output_down")
            if io_resample["mask_up"] != "nearest":
                raise ValueError(f"sampling {sid}: mask_up must be nearest to keep the mask binary")
            bad = {v for v in io_resample.values() if v not in RESAMPLE}
            if bad:
                raise ValueError(f"sampling {sid}: unknown resample filter(s) {sorted(bad)}")
        out[sid] = Sampling(
            id=sid,
            seed=int(s["seed"]),
            generator_device=s["generator_device"],
            resolution=(int(h), int(w)),
            per_variant=s["per_variant"],
            io_resample=io_resample,
        )
    return out


@dataclass(frozen=True)
class Datasets:
    root: Path
    tests: dict  # name -> expected image count
    shadow_dir: str
    mask_dir: str
    gt_dir: str
    mask_threshold: int

    def dir(self, test, part):
        sub = {"shadow": self.shadow_dir, "mask": self.mask_dir, "gt": self.gt_dir}[part]
        return self.root / test / sub


def load_datasets(path=CONFIG_DIR / "datasets.yaml"):
    raw = _read_yaml(path)
    root = Path(raw["root"])
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return Datasets(
        root=root,
        tests={k: int(v) for k, v in raw["tests"].items()},
        shadow_dir=raw["subdirs"]["shadow"],
        mask_dir=raw["subdirs"]["mask"],
        gt_dir=raw["subdirs"]["gt"],
        mask_threshold=int(raw["mask_threshold"]),
    )


@dataclass(frozen=True)
class Run:
    """One (variant, prompt, sampling) output directory."""

    variant: Variant
    prompt_id: str
    prompt_text: str
    sampling: Sampling


@dataclass(frozen=True)
class Config:
    models: ModelsConfig
    prompts: dict
    samplings: dict
    datasets: Datasets

    def prompts_for(self, variant_name):
        return self.prompts[self.models.variants[variant_name].kind]

    def runs(self, variant_name, prompt_ids=None, sampling_ids=None):
        variant = self.models.variants[variant_name]
        prompts = self.prompts_for(variant_name)
        pids = list(prompts) if prompt_ids is None else [p for p in prompt_ids if p in prompts]
        sids = list(self.samplings) if sampling_ids is None else list(sampling_ids)
        # A sampling ID applies only to the variants it defines parameters for.
        sids = [s for s in sids if variant_name in self.samplings[s].per_variant]
        return [Run(variant, pid, prompts[pid], self.samplings[sid]) for pid in pids for sid in sids]


def load_config(config_dir=CONFIG_DIR):
    config_dir = Path(config_dir)
    return Config(
        models=load_models(config_dir / "models.yaml"),
        prompts=load_prompts(config_dir / "prompts.yaml"),
        samplings=load_sampling(config_dir / "sampling.yaml"),
        datasets=load_datasets(config_dir / "datasets.yaml"),
    )

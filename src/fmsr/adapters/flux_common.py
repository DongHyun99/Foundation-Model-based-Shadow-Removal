"""FLUX staged load: text encoders and the fp8-storage transformer are never on the GPU together.

Stage 1 loads CLIP + T5 (bf16), encodes every prompt once, and deletes them.
Stage 2 loads the transformer, enables layerwise casting (fp8 storage, bf16 compute; diffusers default
skip patterns keep norms/embeddings in bf16) and moves the denoising pipeline to the GPU.
"""

import gc
from collections import Counter

import torch

from fmsr.adapters.base import Adapter

MAX_SEQUENCE_LENGTH = 512


def dtype_param_counts(module):
    counts = Counter()
    for p in module.parameters():
        counts[str(p.dtype).removeprefix("torch.")] += p.numel()
    return dict(counts)


class FluxAdapter(Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeds = {}
        self.encoder = None
        tq = self.variant.quantization["transformer"]
        if tq["method"] != "layerwise_casting":
            raise ValueError(f"{self.variant.name}: unsupported transformer quantization {tq}")
        self.storage_dtype = getattr(torch, tq["storage_dtype"])
        self.compute_dtype = getattr(torch, tq["compute_dtype"])

    def _from_pretrained(self, **kwargs):
        return self.pipeline_class.from_pretrained(
            self.variant.hf_repo, revision=self.revision, torch_dtype=torch.bfloat16, **kwargs
        )

    def load(self):
        self.encoder = self._from_pretrained(transformer=None, vae=None)
        self.encoder.text_encoder.to(self.device)
        self.encoder.text_encoder_2.to(self.device)

    def prepare_prompts(self, prompt_map):
        self.on_phase("encode")
        with torch.no_grad():
            for pid, text in prompt_map.items():
                pe, ppe, _ = self.encoder.encode_prompt(
                    prompt=text, prompt_2=text, device=self.device, max_sequence_length=MAX_SEQUENCE_LENGTH
                )
                self.embeds[pid] = (pe.to(self.device, torch.bfloat16), ppe.to(self.device, torch.bfloat16))
        del self.encoder
        self.encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self._load_denoiser()

    def _load_denoiser(self):
        from diffusers import FluxTransformer2DModel

        self.on_phase("load_transformer")
        transformer = FluxTransformer2DModel.from_pretrained(
            self.variant.hf_repo, subfolder="transformer", revision=self.revision, torch_dtype=torch.bfloat16
        )
        transformer.enable_layerwise_casting(storage_dtype=self.storage_dtype, compute_dtype=self.compute_dtype)
        self.pipe = self._from_pretrained(
            transformer=transformer, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None
        ).to(self.device)
        if self.pipe._execution_device.type != "cuda":
            raise RuntimeError(f"{self.variant.name}: execution device is {self.pipe._execution_device}, expected cuda")
        if self.pipe.text_encoder is not None or self.pipe.text_encoder_2 is not None:
            raise RuntimeError(f"{self.variant.name}: text encoders must not be resident during denoising")
        self.pipe.set_progress_bar_config(disable=True)
        self._track_preprocess()
        self.on_phase("generate")

    def _embeds(self, prompt_id, params):
        if params.get("max_sequence_length", MAX_SEQUENCE_LENGTH) != MAX_SEQUENCE_LENGTH:
            raise ValueError(f"prompts were encoded with max_sequence_length={MAX_SEQUENCE_LENGTH}")
        return self.embeds[prompt_id]

    def describe(self):
        counts = dtype_param_counts(self.pipe.transformer)
        storage = str(self.storage_dtype).removeprefix("torch.")
        if not counts.get(storage):
            raise RuntimeError(f"{self.variant.name}: no {storage} parameters after layerwise casting: {counts}")
        quant = dict(self.variant.quantization)
        quant["transformer"] = {**quant["transformer"], "param_counts_by_dtype": counts}
        return {
            "pipeline_class": self.pipeline_class.__name__,
            "hf_revision": self.revision,
            "quantization": quant,
            "text_encoder_handling": "flux_stage1_deleted",
        }

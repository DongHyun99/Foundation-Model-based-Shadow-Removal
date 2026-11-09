"""Qwen-Image nf4 loading: per-component bitsandbytes configs (diffusers maintainer recipe).

The transformer keeps transformer_blocks.0.img_mod unquantized; the text encoder (Qwen2.5-VL) uses the
transformers config with no skip list so its default lm_head skip is preserved.
"""

from collections import Counter

import torch

from fmsr.adapters.base import Adapter

IMG_MOD_SKIP = "transformer_blocks.0.img_mod"


def _quant_config(quantization):
    import diffusers
    import transformers
    from diffusers import PipelineQuantizationConfig

    tq, eq = quantization["transformer"], quantization["text_encoder"]
    if tq["backend"] != "bitsandbytes_4bit" or eq["backend"] != "transformers_bitsandbytes_4bit":
        raise ValueError(f"unsupported Qwen quantization: {quantization}")
    return PipelineQuantizationConfig(
        quant_mapping={
            "transformer": diffusers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=tq["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(torch, tq["bnb_4bit_compute_dtype"]),
                llm_int8_skip_modules=list(tq["llm_int8_skip_modules"]),
            ),
            "text_encoder": transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=eq["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(torch, eq["bnb_4bit_compute_dtype"]),
            ),
        }
    )


def linear_counts(module):
    import bitsandbytes as bnb

    counts = Counter()
    for m in module.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            counts["linear4bit"] += 1
        elif isinstance(m, torch.nn.Linear):
            counts["linear_bf16"] += 1
    return dict(counts)


class QwenAdapter(Adapter):
    def load(self):
        self.pipe = self.pipeline_class.from_pretrained(
            self.variant.hf_repo,
            revision=self.revision,
            quantization_config=_quant_config(self.variant.quantization),
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        if self.pipe._execution_device.type != "cuda":
            raise RuntimeError(f"{self.variant.name}: execution device is {self.pipe._execution_device}, expected cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self._track_preprocess()

    def prepare_prompts(self, prompt_map):
        # Qwen encodes the (short) prompt per call with the resident nf4 text encoder.
        self.on_phase("generate")

    def describe(self):
        import bitsandbytes as bnb

        img_mod = self.pipe.transformer.get_submodule(IMG_MOD_SKIP)
        if any(isinstance(m, bnb.nn.Linear4bit) for m in img_mod.modules()):
            raise RuntimeError(f"{self.variant.name}: {IMG_MOD_SKIP} must stay unquantized")
        quant = dict(self.variant.quantization)
        quant["transformer"] = {**quant["transformer"], "linear_counts": linear_counts(self.pipe.transformer)}
        quant["text_encoder"] = {**quant["text_encoder"], "linear_counts": linear_counts(self.pipe.text_encoder)}
        if not quant["transformer"]["linear_counts"].get("linear4bit") or not quant["text_encoder"]["linear_counts"].get("linear4bit"):
            raise RuntimeError(f"{self.variant.name}: nf4 quantization not applied: {quant}")
        return {
            "pipeline_class": self.pipeline_class.__name__,
            "hf_revision": self.revision,
            "quantization": quant,
            "text_encoder_handling": "nf4_resident",
        }

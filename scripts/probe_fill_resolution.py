"""Diagnostic: is FLUX.1-Fill-dev's failure at 256x256 caused by fp8 or by resolution?

Usage: CUDA_VISIBLE_DEVICES=<g> venv/bin/python scripts/probe_fill_resolution.py {default|skipio|cpuoffload}
Runs one AISTD image at 256 and 512 and prints the output sanity metrics (see reports/fill_resolution_probe.json).
"""

import json
import sys

import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
from fmsr.config import load_config
from fmsr.data import load_rgb, load_mask_bin
from fmsr.sanity import check_output

repo = "black-forest-labs/FLUX.1-Fill-dev"
rev = json.load(open("reports/download.json"))["FLUX.1-Fill-dev"]["revision"]
ds = load_config().datasets
img = load_rgb(ds.dir("AISTD", "shadow") / "100-1.png")
mask = load_mask_bin(ds.dir("AISTD", "mask") / "100-1.png", 127)

enc = FluxFillPipeline.from_pretrained(repo, revision=rev, transformer=None, vae=None, torch_dtype=torch.bfloat16)
enc.text_encoder.to("cuda"); enc.text_encoder_2.to("cuda")
with torch.no_grad():
    pe, ppe, _ = enc.encode_prompt(prompt="remove the shadow", prompt_2="remove the shadow", device="cuda", max_sequence_length=512)
del enc; torch.cuda.empty_cache()

tx = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", revision=rev, torch_dtype=torch.bfloat16)
mode = sys.argv[1]
if mode == "default":
    tx.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
elif mode == "skipio":
    tx.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
                                skip_modules_pattern=DEFAULT_SKIP_MODULES_PATTERN + ("x_embedder", "context_embedder", "proj_out"))
pipe = FluxFillPipeline.from_pretrained(repo, revision=rev, transformer=tx, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
if mode == "cpuoffload":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
for res in [256, 512]:
    out = pipe(image=img.resize((res, res)), mask_image=mask.resize((res, res)), prompt_embeds=pe, pooled_prompt_embeds=ppe,
               height=res, width=res, guidance_scale=30.0, num_inference_steps=50,
               generator=torch.Generator("cpu").manual_seed(0)).images[0]
    out.save(f"/tmp/fill_{mode}_{res}.png")
    print(mode, res, check_output(out.resize((256, 256)), img, mask), flush=True)

# Foundation-Model-based Shadow Removal

Zero-shot shadow removal with image-generation foundation models: **no training or fine-tuning**, only
prompt + image (+ mask). Four variants are benchmarked on the AISTD, SRD and WSRD test sets:

| Variant (output dir) | Checkpoint | Pipeline (diffusers 0.42.1) | Mask use |
|---|---|---|---|
| `FLUX.1-Fill-dev` | `black-forest-labs/FLUX.1-Fill-dev` | `FluxFillPipeline` | model input (inpaint) |
| `FLUX.1-Kontext-dev` | `black-forest-labs/FLUX.1-Kontext-dev` | `FluxKontextPipeline` | compositing only (edit) |
| `Qwen-Image-Inpaint` | `Qwen/Qwen-Image` | `QwenImageInpaintPipeline` (strength 1.0) | model input (inpaint) |
| `Qwen-Image-Edit-2509` | `Qwen/Qwen-Image-Edit-2509` | `QwenImageEditPlusPipeline` | compositing only (edit) |

Every variant saves both the **raw** model output and a **composite**
`mask * raw + (1 - mask) * input` (pixels outside the shadow mask are byte-identical to the input).

## Setup

```bash
virtualenv -p python3.12 venv
venv/bin/pip install -r requirements.txt   # exact pins; torch from the cu128 index
venv/bin/pip install -e .
venv/bin/python scripts/check_access.py    # HF token + gated FLUX access
venv/bin/python scripts/download.py        # diffusers components, revisions -> reports/download.json
```

Datasets are expected at `../dataset/{AISTD,SRD,WSRD}/test_{A,B,C}` (shadow / mask / ground truth;
configured in `configs/datasets.yaml`). All images are 256x256.

## Running

```bash
venv/bin/python -m pytest -q tests          # CPU unit tests
venv/bin/python scripts/smoke.py --sanity-waiver 'FLUX.1-Fill-dev=<reason>'   # gates, resume, steal drill
bash scripts/launch_all.sh                  # full run: one supervisor per GPU (8 GPUs, 2 per variant)
venv/bin/python scripts/verify_counts.py --expect-complete      # -> reports/counts.json
venv/bin/python scripts/evaluate.py --require-complete          # -> results/*.csv
```

- **Resume:** a task is done when both its raw and composite PNGs exist. Re-running any command skips
  finished work; a worker whose variant is finished exits before loading a model.
- **Scheduling:** GPUs claim images through atomic lock files under `$OUT_ROOT/.claims`. When a GPU's own
  variant runs out of claimable work, its supervisor relaunches it on the variant with the most remaining
  weighted work (`cost_weight`, `min_remaining` in `configs/models.yaml`). Each process loads exactly one
  model. After a crash, stale locks are reclaimed; a stale-reclaim race can at most duplicate one
  deterministic generation (outputs stay identical). Failed images leave `.failed` markers and are retried
  on the next launch.
- **Detached runs:** start long jobs inside `tmux` so they survive the launching shell.
- Overrides (used by the smoke drill): `OUT_ROOT`, `PROMPTS`, `SAMPLINGS`, `LIMIT_PER_DATASET`,
  `MIN_REMAINING`, `GPU_MAP="0:FLUX.1-Fill-dev 1:..."`, `STAGGER_S`, `FOREGROUND=1`.

## Output layout

```
outputs/<Variant>/<promptID>/<samplingID>/{raw,composite}/<Test>/<test_A stem>.png
outputs/<Variant>/<promptID>/<samplingID>/run_config.yaml          # committed
outputs/<Variant>/run_summary_<host>_gpu<g>_<startUTC>.json         # committed (per process)
```

Images, weights and `venv/` are git-ignored; run configs, summaries, reports and CSVs are committed.
SRD files keep their full stem (`_MG_5580.JPG.jpg` -> `_MG_5580.JPG.png`); evaluation matches files by
key with all image extensions stripped, like `../measure_lpips.py`.

## Configuration

- `configs/prompts.yaml` - fixed prompt IDs. Inpaint: `i0` `""`, `i1` `remove the shadow`,
  `i2` `the same surface without any shadow, natural even lighting, consistent texture and color`,
  `i3` `shadow-free, uniformly lit, photorealistic`. Edit: `e1` `Remove the shadow.`,
  `e2` `Remove all shadows from the image while keeping everything else exactly the same.`,
  `e3` `Relight the shadowed region so that it matches the surrounding illumination, preserving texture and color.`,
  `e4` `Make the image shadow-free with uniform lighting.` Never edit an existing ID; add a new one.
- `configs/sampling.yaml` - sampling ID `default`: seed 0 (fresh CPU generator per image), 256x256.
  Fill 50 steps / guidance 30; Kontext 28 steps / guidance 2.5; Qwen-Inpaint 50 steps / true CFG 4.0 /
  strength 1.0 / negative `" "`; Qwen-Edit-2509 40 steps / true CFG 4.0 / guidance 1.0 / negative `" "`.
  Other resolutions or seeds are added as new sampling IDs.
- `run_config.yaml` records model revision, applied quantization (with verified layer/parameter counts),
  prompt, sampling, resize overrides, mask threshold and package versions. Relaunching with different
  reproducibility-relevant values aborts instead of mixing results.

### Quantization (one model per 24 GB RTX A5000)

- **FLUX (Fill, Kontext):** transformer weights stored in fp8 (`float8_e4m3fn`) with bf16 compute via
  diffusers `enable_layerwise_casting`; norms and embeddings stay bf16 (diffusers default skip patterns).
  Ampere has no fp8 compute, so this saves memory only. CLIP and T5-XXL run in bf16 in a first stage
  that encodes all prompts once and is then deleted, so text encoders and transformer never share the GPU.
- **Qwen (Inpaint, Edit-2509):** transformer and Qwen2.5-VL text encoder in bitsandbytes nf4 (bf16
  compute); `transformer_blocks.0.img_mod` and the text encoder's `lm_head` stay bf16.
- Measured NVML peaks: Fill 15.6 GB, Kontext 15.7 GB, Qwen-Inpaint 18.3 GB, Qwen-Edit-2509 18.3 GB.

### Forcing 256x256 inference

- Kontext: `height=width=256`, `max_area=256*256`, `_auto_resize=False`.
- Qwen-Edit-2509: the pipeline always resizes the conditioning image to ~1 MP (`VAE_IMAGE_SIZE`, no call
  argument). The adapter sets that module global to 256*256 for each call and restores it afterwards;
  the Qwen2.5-VL condition image keeps its default 384*384 area (`CONDITION_IMAGE_SIZE`).
- Every conditioning-image preprocess size is recorded; the smoke test asserts it is (256, 256).

### Known limitation: FLUX.1-Fill-dev at 256x256

At 256x256, FLUX.1-Fill-dev largely ignores the input image (raw output PSNR vs. input in the
non-shadow region 7-13 dB). The same image at 512x512 gives ~29 dB, and a full-bf16 transformer behaves
identically to fp8 (`reports/fill_resolution_probe.json`, `scripts/probe_fill_resolution.py`), so this is
model behaviour at the specified resolution, not quantization. The smoke test records it as an explicit
sanity waiver. Fill's **composite** output is still well defined (the mask region is generated, the rest
is the input).

## Evaluation (Python quick check; MATLAB is authoritative)

`scripts/evaluate.py` scores raw and composite outputs per variant x prompt x sampling x test, plus an
`Input` baseline (shadow image vs. GT).

- **"RMSE" (ISTD / ShadowFormer convention):** per-pixel sum over L, a, b of `|Lab(pred) - Lab(GT)|`
  (`skimage.color.rgb2lab`, D65 by default), averaged over all / shadow (S) / non-shadow (NS) pixels per
  image, then over images. `*_pooled` columns divide the total error by the total pixel count.
- Mask: `gray(test_B) > 127` (SRD masks are JPEG); `--mask-rule nonzero` is available.
- **PSNR / SSIM / MAE** on RGB uint8 (`structural_similarity(..., channel_axis=-1, data_range=255)`).
- **LPIPS** alex and vgg, inputs scaled to [-1, 1] (same as `../measure_lpips.py`).
- Differences from MATLAB evaluation scripts: MATLAB `makecform('srgb2lab')` is D50-adapted
  (`--lab-whitepoint D50` approximates it), and MATLAB scripts may binarize masks differently.
  Sanity check: the AISTD `Input` row (all 8.76 / S 38.1 / NS 2.94) matches the published ISTD+ input
  row within the expected tolerance.

Outputs: `results/metrics_summary.csv` (one row per variant x prompt x sampling x test x output type,
plus `Input` rows) and `results/per_image/<Variant>.csv`, `results/per_image/Input.csv`.

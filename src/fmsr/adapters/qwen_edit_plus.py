from contextlib import contextmanager

from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as edit_plus_module

from fmsr.adapters.qwen_common import QwenAdapter


@contextmanager
def qwen_edit_sizes(vae_area, condition_area):
    """Temporarily set the module globals QwenImageEditPlusPipeline reads at call time.

    The pipeline always resizes the conditioning image to VAE_IMAGE_SIZE area (~1MP) with no call
    argument to change it; patching the global is the only way to keep VAE conditioning at the
    sampling resolution. CONDITION_IMAGE_SIZE is the Qwen2.5-VL input area.
    """
    for name in ("VAE_IMAGE_SIZE", "CONDITION_IMAGE_SIZE"):
        if not hasattr(edit_plus_module, name):
            raise RuntimeError(f"diffusers internals changed: {edit_plus_module.__name__}.{name} missing")
    saved = edit_plus_module.VAE_IMAGE_SIZE, edit_plus_module.CONDITION_IMAGE_SIZE
    edit_plus_module.VAE_IMAGE_SIZE, edit_plus_module.CONDITION_IMAGE_SIZE = vae_area, condition_area
    try:
        yield
    finally:
        edit_plus_module.VAE_IMAGE_SIZE, edit_plus_module.CONDITION_IMAGE_SIZE = saved


class QwenEditPlusAdapter(QwenAdapter):
    """Qwen-Image-Edit-2509: instruction editing; the mask is only used for compositing by the runner."""

    pipeline_class = QwenImageEditPlusPipeline

    def resize_overrides(self, params, resolution):
        return {"VAE_IMAGE_SIZE": params["vae_image_area"], "CONDITION_IMAGE_SIZE": params["condition_area"]}

    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        h, w = resolution
        if params["vae_image_area"] != h * w:
            raise ValueError(f"Qwen-Edit VAE conditioning must match {h}x{w}: got area {params['vae_image_area']}")
        with qwen_edit_sizes(params["vae_image_area"], params["condition_area"]):
            out = self.pipe(
                image=[image],
                prompt=prompt_text,
                negative_prompt=params["negative_prompt"],
                true_cfg_scale=params["true_cfg_scale"],
                guidance_scale=params["guidance_scale"],
                num_inference_steps=params["num_inference_steps"],
                height=h,
                width=w,
                generator=generator,
            ).images[0]
        return self._check_output(out, resolution)

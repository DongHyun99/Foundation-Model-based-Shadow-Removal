from diffusers import QwenImageInpaintPipeline

from fmsr.adapters.qwen_common import QwenAdapter


class QwenInpaintAdapter(QwenAdapter):
    """Qwen-Image text-to-image weights with mask-based inpainting (white = repaint)."""

    pipeline_class = QwenImageInpaintPipeline

    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        h, w = resolution
        out = self.pipe(
            prompt=prompt_text,
            negative_prompt=params["negative_prompt"],
            image=image,
            mask_image=mask_l,
            strength=params["strength"],
            true_cfg_scale=params["true_cfg_scale"],
            num_inference_steps=params["num_inference_steps"],
            height=h,
            width=w,
            generator=generator,
        ).images[0]
        return self._check_output(out, resolution)

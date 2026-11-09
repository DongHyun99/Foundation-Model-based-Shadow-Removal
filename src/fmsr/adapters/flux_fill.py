from diffusers import FluxFillPipeline

from fmsr.adapters.flux_common import FluxAdapter


class FluxFillAdapter(FluxAdapter):
    """FLUX.1-Fill-dev: true inpainting; the binary mask (white = repaint) is a model input."""

    pipeline_class = FluxFillPipeline

    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        pe, ppe = self._embeds(prompt_id, params)
        h, w = resolution
        out = self.pipe(
            image=image,
            mask_image=mask_l,
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            height=h,
            width=w,
            guidance_scale=params["guidance_scale"],
            num_inference_steps=params["num_inference_steps"],
            max_sequence_length=params["max_sequence_length"],
            generator=generator,
        ).images[0]
        return self._check_output(out, resolution)

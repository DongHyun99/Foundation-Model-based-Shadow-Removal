from diffusers import FluxKontextInpaintPipeline

from fmsr.adapters.flux_kontext import FluxKontextAdapter


class FluxKontextInpaintAdapter(FluxKontextAdapter):
    """FLUX.1-Kontext-dev weights with the binary mask as a model input (white = repaint).

    The full shadow image is the in-context condition; the mask region is denoised from `strength` and the
    rest is blended back to the image latents every step.
    """

    pipeline_class = FluxKontextInpaintPipeline

    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        pe, ppe = self._embeds(prompt_id, params)
        h, w = resolution
        if params["max_area"] != h * w or params["_auto_resize"]:
            raise ValueError(f"Kontext must run at {h}x{w}: got max_area={params['max_area']}, _auto_resize={params['_auto_resize']}")
        out = self.pipe(
            image=image,
            mask_image=mask_l,
            strength=params["strength"],
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            height=h,
            width=w,
            max_area=params["max_area"],
            _auto_resize=params["_auto_resize"],
            guidance_scale=params["guidance_scale"],
            num_inference_steps=params["num_inference_steps"],
            max_sequence_length=params["max_sequence_length"],
            generator=generator,
        ).images[0]
        return self._check_output(out, resolution)

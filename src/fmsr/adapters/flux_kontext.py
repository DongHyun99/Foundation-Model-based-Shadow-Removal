from diffusers import FluxKontextPipeline

from fmsr.adapters.flux_common import FluxAdapter


class FluxKontextAdapter(FluxAdapter):
    """FLUX.1-Kontext-dev: instruction editing; the mask is only used for compositing by the runner.

    Auto-resize is disabled and max_area is pinned to the sampling resolution so inference runs at that size.
    """

    pipeline_class = FluxKontextPipeline

    def resize_overrides(self, params, resolution):
        return {"_auto_resize": params["_auto_resize"], "max_area": params["max_area"]}

    def generate(self, image, mask_l, prompt_id, prompt_text, params, resolution, generator):
        pe, ppe = self._embeds(prompt_id, params)
        h, w = resolution
        if params["max_area"] != h * w or params["_auto_resize"]:
            raise ValueError(f"Kontext must run at {h}x{w}: got max_area={params['max_area']}, _auto_resize={params['_auto_resize']}")
        out = self.pipe(
            image=image,
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

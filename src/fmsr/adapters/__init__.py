"""Adapter registry. Imports are lazy so each worker only imports its own pipeline."""

import importlib

ADAPTERS = {
    "flux_fill": ("fmsr.adapters.flux_fill", "FluxFillAdapter"),
    "flux_kontext": ("fmsr.adapters.flux_kontext", "FluxKontextAdapter"),
    "qwen_inpaint": ("fmsr.adapters.qwen_inpaint", "QwenInpaintAdapter"),
    "qwen_edit_plus": ("fmsr.adapters.qwen_edit_plus", "QwenEditPlusAdapter"),
}


def get_adapter_class(name):
    module, cls = ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)

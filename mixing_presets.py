import os

PRESETS = {
    "fractals": "/path/to/mixingsets/fractals",
    "pixmix_fractals": "/path/to/mixingsets/fractals",
    "pixmix_fvis": "/path/to/mixingsets/first_layers_resized256_onevis",
    "diffusemix_best": "/path/to/mixingsets/diffusemix_fractal_library",
    "diffusemix_generated_bank": "/path/to/mixingsets/diffusemix_generated_bank",
    "layermix_best": "/path/to/mixingsets/layermix_fractals_gray",
    "layermix_fractals_gray": "/path/to/mixingsets/layermix_fractals_gray",
    "ipmix_best": "/path/to/mixingsets/ipmix_set",
    "ipmix_set": "/path/to/mixingsets/ipmix_set",
}

DEFAULT_PRESETS = {
    "diffusemix": "diffusemix_best",
    "layermix": "layermix_best",
    "ipmix": "ipmix_best",
}

PRESET_ERROR_HINTS = {
    "layermix_best": (
        "Expected a gray-scale fractal set for LayerMix. "
        "Generate or copy the gray fractal library to this path, "
        "or pass --mixing-set with the correct directory."
    ),
    "ipmix_best": (
        "Expected the official IPMix mixing set. "
        "Place it at this path or pass --mixing-set explicitly."
    ),
    "diffusemix_best": (
        "Expected the DiffuseMix fractal library (paper default). "
        "Download the fractal dataset from the DiffuseMix repo and place it here, "
        "or pass --mixing-set/--diffusemix-fractal-set explicitly."
    ),
    "diffusemix_generated_bank": (
        "Expected a pre-generated DiffuseMix image bank. "
        "Generate with the diffusion pipeline and place it here."
    ),
    "ipmix_set": (
        "Expected the official IPMix mixing set. "
        "Place it at this path or pass --mixing-set explicitly."
    ),
    "layermix_fractals_gray": (
        "Expected a gray-scale fractal set for LayerMix. "
        "Generate or copy the gray fractal library to this path."
    ),
}


def resolve_preset(name, require_exists=True):
    path = PRESETS.get(name, name)
    if require_exists and not os.path.isdir(path):
        hint = PRESET_ERROR_HINTS.get(name)
        msg = f"Mixing preset '{name}' resolved to '{path}', but the path does not exist."
        if hint:
            msg = f"{msg} {hint}"
        raise ValueError(msg)
    return path
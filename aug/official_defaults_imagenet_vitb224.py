TIMM_VERSION = "0.9.16"

OFFICIAL_DEFAULTS = {
    "augmix": {
        "mixture_width": 3,
        "mixture_depth": -1,
        "aug_severity": 1,
        "alpha": 1.0,
        "source": (
            "google-research/augmix/imagenet.py "
            "(args: mixture_width=3, mixture_depth=-1, aug_severity=1, aug_prob_coeff=1.0)"
        ),
    },
    "cutout": {
        "enabled": False,
        "size": 48,
        "source": (
            "uoguelph-mlrg/Cutout has no ImageNet recipe; "
            "size=48 is practical fallback when enabled."
        ),
    },
    "gridmask": {
        "d_min": 96,
        "d_max": 224,
        "rotate": 360,
        "ratio": 0.6,
        "mode": 1,
        "prob": 0.8,
        "source": "dvlab-research/GridMask/imagenet_grid/imagenet_amp.py",
    },
    "afa": {
        "min_str": 0.0,
        "mean_str": 10.0,
        "freq_cut": 1,
        "phase_cut": 1,
        "granularity": 448,
        "source": (
            "third_party/afa-augment/main.py (ImageNet experiments) + "
            "third_party/afa-augment/config_utils.py (make_config defaults)"
        ),
    },
    "pixmix": {
        "k": 4,
        "beta": 4,
        "aug_severity": 1,
        "all_ops": False,
        "source": "andyzoujm/pixmix/imagenet.py (args: k=4, beta=4, aug_severity=1)",
    },
    "ipmix": {
        "k": 3,
        "t": 3,
        "beta": 4,
        "aug_severity": 1,
        "all_ops": False,
        "source": "hzlsaber/IPMix/imagenet.py (args: k=3, t=3, beta=4, aug_severity=1)",
    },
    "diffusemix": {
        "fractal_lambda": 0.20,
        "alpha": 0.5,
        "beta": 0.5,
        "concat_prob": 0.5,
        "source": "khawar-islam/diffuseMix (paper: lambda=0.20; repo mixes fractal blend in generate step)",
    },
    "layermix": {
        "depth": 3,
        "width": 3,
        "magnitude": 3,
        "blending": 3.0,
        "source": "ahmadmughees/LayerMix/imagenet.py (args: depth=3, width=3, magnitude=3, blending_ratio=3)",
    },
    "mixup": {
        "alpha": 0.0,
        "source": f"timm train.py defaults (timm=={TIMM_VERSION})",
    },
    "cutmix": {
        "alpha": 0.0,
        "source": f"timm train.py defaults (timm=={TIMM_VERSION})",
    },
    "mixup_cutmix_recipe": {
        "imagenet_standard": {"mixup_alpha": 0.8, "cutmix_alpha": 1.0},
        "source": f"optional recipe (not default) for ImageNet; timm=={TIMM_VERSION} compatible",
    },
}

EXTENDED_DEFAULTS = {
    "afa": {
        "phase_min": 0.0,
        "phase_max": 1.0,
        "per_channel": True,
        "source": "local extension (non-official) for our AFA implementation knobs",
    },
}


def get_defaults(mode=None):
    """Return OFFICIAL defaults. If mode is provided, return that method's defaults."""
    if mode:
        return OFFICIAL_DEFAULTS.get(mode)
    return OFFICIAL_DEFAULTS.copy()


def get_extended_defaults(mode=None):
    """Return EXTENDED (non-official) defaults. If mode is provided, return that method's defaults."""
    if mode:
        return EXTENDED_DEFAULTS.get(mode)
    return EXTENDED_DEFAULTS.copy()

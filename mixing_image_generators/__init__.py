from .base import BaseGenerator
from .moire import MoireGenerator
from .coloredfractal import FractalGenerator
from .colorbackground import ColorBackgroundGenerator
from .coloredmoire import ColoredMoireGenerator
from .deadleaves import DeadLeavesGenerator
from .perlin import PerlinNoiseGenerator
from .stripe import StripeGenerator
from .fourier2019 import FourierBasis2019Generator
from .afa import AFAGenerator

__all__ = [
    'create_generator',
    'BaseGenerator',
    'MoireGenerator',
    'FractalGenerator',
    'ColorBackgroundGenerator',
    'ColoredMoireGenerator',
    'DeadLeavesGenerator',
    'PerlinNoiseGenerator',
    'StripeGenerator',
    'FourierBasis2019Generator',
    'AFAGenerator',
]

def create_generator(name: str, **kwargs) -> BaseGenerator:
    """
    Factory function to instantiate generators by name.

    Args:
        name (str): 'moire' or 'fractal'
        **kwargs: Dictionary of arguments (usually argparse args) to be passed
                  to the generator constructor. Generators should handle
                  irrelevant args gracefully (e.g. using **kwargs in __init__).

    Returns:
        BaseGenerator: An instance of the requested generator.
    """
    name = name.lower()
    
    if name == 'moire':
        return MoireGenerator(**kwargs)
    elif name == 'fractal':
        return FractalGenerator(**kwargs)
    elif name in ('colorbackground', 'bg', 'background'):
        return ColorBackgroundGenerator(**kwargs)
    elif name == 'coloredmoire':
        return ColoredMoireGenerator(**kwargs)
    elif name == 'deadleaves':
        return DeadLeavesGenerator(**kwargs)
    elif name == 'perlin':
        return PerlinNoiseGenerator(**kwargs)
    elif name == 'stripe':
        return StripeGenerator(**kwargs)
    elif name == 'fourier2019':
        return FourierBasis2019Generator(**kwargs)
    elif name == 'afa':
        return AFAGenerator(**kwargs)
    else:
        raise ValueError(
            f"Unknown generator backend: {name}. Supported: 'moire', 'fractal', "
            "'colorbackground', 'coloredmoire', 'deadleaves', 'perlin', 'stripe', "
            "'fourier2019', 'afa'"
        )

"""On-the-fly Fourier stripe generator."""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
from PIL import Image

from .base import BaseGenerator


class StripeGenerator(BaseGenerator):
    """Generate a single plane-wave stripe pattern for mixing."""

    def __init__(
        self,
        size: int = 224,
        online_stripe_freq_min: float = 1.0,
        online_stripe_freq_max: float = 100.0,
        online_stripe_amp: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.freq_min = float(online_stripe_freq_min)
        self.freq_max = float(online_stripe_freq_max)
        self.amp = float(online_stripe_amp)
        self.rng = random.Random()
        self.last_info = None
        u = np.linspace(0.0, 1.0, self.size, endpoint=False, dtype=np.float32)
        v = np.linspace(0.0, 1.0, self.size, endpoint=False, dtype=np.float32)
        self.u, self.v = np.meshgrid(u, v, indexing="xy")

    def generate(self) -> Image.Image:
        freq = self.rng.uniform(self.freq_min, self.freq_max)
        theta = self.rng.uniform(0.0, math.pi)
        phi = self.rng.uniform(0.0, 2.0 * math.pi)
        self.last_info = {
            "freq": float(freq),
            "theta": float(theta),
            "phi": float(phi),
            "amp": float(self.amp),
        }
        phase = 2.0 * math.pi * freq * (self.u * math.cos(theta) + self.v * math.sin(theta)) + phi
        stripe = 0.5 + self.amp * np.sin(phase)
        stripe = np.clip(stripe, 0.0, 1.0)
        rgb = np.repeat(stripe[..., None], 3, axis=-1)
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

import math
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

from .base import BaseGenerator
from .coloredfractal import bg_solid, bg_linear, bg_softnoise


BG_CHOICES = [bg_solid, bg_linear, bg_softnoise]


def random_color(rng: random.Random) -> Tuple[int, int, int]:
    h = rng.random()
    s = rng.uniform(0.5, 0.95)
    v = rng.uniform(0.85, 1.0)
    i = int(h * 6) % 6
    f = h * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    V = int(255 * v)
    if i == 0:
        return (V, t, p)
    if i == 1:
        return (q, V, p)
    if i == 2:
        return (p, V, t)
    if i == 3:
        return (p, q, V)
    if i == 4:
        return (t, p, V)
    return (V, p, q)


class ColoredMoireGenerator(BaseGenerator):
    """
    On-the-fly colored Moire generator with layered alpha compositing.
    """
    def __init__(
        self,
        size: int = 224,
        online_moire_freq_min: int = 1,
        online_moire_freq_max: int = 100,
        online_moire_centers_min: int = 1,
        online_moire_centers_max: int = 3,
        online_moire_margin: float = 0.08,
        online_coloredmoire_layers_min: int = 1,
        online_coloredmoire_layers_max: int = 2,
        online_coloredmoire_alpha_pow: float = 1.15,
        online_coloredmoire_bg_mode: str = "random",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        fmin = int(online_moire_freq_min)
        fmax = int(online_moire_freq_max)
        self.freq_list: List[int] = list(range(fmin, fmax + 1))
        self.centers_min = int(online_moire_centers_min)
        self.centers_max = int(online_moire_centers_max)
        self.margin = float(self.size) * float(online_moire_margin)
        self.layers_min = max(1, int(online_coloredmoire_layers_min))
        self.layers_max = max(self.layers_min, int(online_coloredmoire_layers_max))
        self.alpha_pow = float(online_coloredmoire_alpha_pow)
        self.bg_mode = str(online_coloredmoire_bg_mode).lower()
        self.rng = random.Random()
        self.last_info = None

        y = np.arange(self.size, dtype=np.float32)
        x = np.arange(self.size, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x, y, indexing='xy')
        self.scale = (2.0 * math.pi) / float(self.size)

    def _moire_gray(self) -> Image.Image:
        centers = self.rng.randint(self.centers_min, self.centers_max)
        if centers <= len(self.freq_list):
            freqs = self.rng.sample(self.freq_list, k=centers)
        else:
            freqs = [self.rng.choice(self.freq_list) for _ in range(centers)]
        z = np.zeros((self.size, self.size), dtype=np.float32)
        for f in freqs:
            cx = self.rng.uniform(self.margin, self.size - self.margin)
            cy = self.rng.uniform(self.margin, self.size - self.margin)
            dx = self.xx - cx
            dy = self.yy - cy
            r = np.sqrt(dx * dx + dy * dy)
            z += np.sin(self.scale * f * r)
        z /= float(centers)
        zmin, zmax = z.min(), z.max()
        if zmax > zmin + 1e-12:
            zn = (z - zmin) / (zmax - zmin)
            img_arr = (zn * 255.0 + 0.5).astype(np.uint8)
        else:
            img_arr = np.zeros_like(z, dtype=np.uint8)
        return Image.fromarray(img_arr, mode='L')

    def _background(self) -> Image.Image:
        if self.bg_mode == "solid":
            return bg_solid(self.size, self.rng)
        if self.bg_mode == "linear":
            return bg_linear(self.size, self.rng)
        if self.bg_mode == "softnoise":
            return bg_softnoise(self.size, self.rng)
        return self.rng.choice(BG_CHOICES)(self.size, self.rng)

    def generate(self) -> Image.Image:
        base_bg = self._background()
        canvas = base_bg.convert('RGBA')
        layers = self.rng.randint(self.layers_min, self.layers_max)
        self.last_info = {
            "layers": int(layers),
            "centers_min": int(self.centers_min),
            "centers_max": int(self.centers_max),
            "freq_min": int(min(self.freq_list)),
            "freq_max": int(max(self.freq_list)),
            "alpha_pow": float(self.alpha_pow),
            "bg_mode": str(self.bg_mode),
        }
        for _ in range(layers):
            gray = self._moire_gray()
            color1 = random_color(self.rng)
            color2 = random_color(self.rng)
            if self.rng.random() < 0.5:
                color1, color2 = color2, color1
            rgb = ImageOps.colorize(gray, black=color1, white=color2)
            alpha = ImageOps.autocontrast(gray)
            a = np.asarray(alpha, dtype=np.float32) / 255.0
            a = np.power(a, self.alpha_pow)
            a = (a * 255.0 + 0.5).astype(np.uint8)
            rgba = rgb.convert('RGBA')
            rgba.putalpha(Image.fromarray(a, mode='L'))
            canvas = Image.alpha_composite(canvas, rgba)
        return canvas.convert('RGB')

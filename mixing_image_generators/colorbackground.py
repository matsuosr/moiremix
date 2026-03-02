import random
from PIL import Image, ImageFilter
import numpy as np

from .base import BaseGenerator


def _hsv_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    h6 = (h * 6.0) % 6.0
    c = v * s
    x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
    m = v - c
    zeros = np.zeros_like(h6)
    r = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5],
        [c, x, zeros, zeros, x],
        default=c,
    )
    g = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5],
        [x, c, c, x, zeros],
        default=zeros,
    )
    b = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5],
        [zeros, zeros, x, c, c],
        default=x,
    )
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _diamond_square(size: int, rng: random.Random) -> np.ndarray:
    n = 1
    while n < size - 1:
        n *= 2
    grid = np.zeros((n + 1, n + 1), dtype=np.float32)
    grid[0, 0] = rng.random()
    grid[0, n] = rng.random()
    grid[n, 0] = rng.random()
    grid[n, n] = rng.random()
    step = n
    scale = 1.0
    while step > 1:
        half = step // 2
        for y in range(half, n, step):
            for x in range(half, n, step):
                avg = (
                    grid[y - half, x - half]
                    + grid[y - half, x + half]
                    + grid[y + half, x - half]
                    + grid[y + half, x + half]
                ) * 0.25
                grid[y, x] = avg + rng.uniform(-scale, scale)
        for y in range(0, n + 1, half):
            for x in range((y + half) % step, n + 1, step):
                vals = []
                if y - half >= 0:
                    vals.append(grid[y - half, x])
                if y + half <= n:
                    vals.append(grid[y + half, x])
                if x - half >= 0:
                    vals.append(grid[y, x - half])
                if x + half <= n:
                    vals.append(grid[y, x + half])
                avg = sum(vals) / float(len(vals))
                grid[y, x] = avg + rng.uniform(-scale, scale)
        step = half
        scale *= 0.5
    grid -= grid.min()
    denom = grid.max() + 1e-8
    grid = grid / denom
    if grid.shape[0] != size:
        img = Image.fromarray((grid * 255.0 + 0.5).astype(np.uint8), mode='L')
        img = img.resize((size, size), Image.BILINEAR)
        grid = np.asarray(img, dtype=np.float32) / 255.0
    return grid


class ColorBackgroundGenerator(BaseGenerator):
    """
    On-the-fly color background generator using diamond-square terrain.
    """
    def __init__(
        self,
        size: int = 224,
        online_bg_use_base_size: bool = True,
        online_bg_base_size: int = 129,
        online_bg_blur_radius: float = 0.8,
        online_bg_gamma_min: float = 0.4,
        online_bg_gamma_max: float = 0.8,
        online_bg_sat_min: float = 0.3,
        online_bg_sat_max: float = 1.0,
        online_bg_val_min: float = 0.5,
        online_bg_val_max: float = 1.0,
        online_bg_hue_offset: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.use_base_size = bool(online_bg_use_base_size)
        self.base_size = int(online_bg_base_size)
        self.blur_radius = float(online_bg_blur_radius)
        self.gamma_min = float(online_bg_gamma_min)
        self.gamma_max = float(online_bg_gamma_max)
        self.sat_min = float(online_bg_sat_min)
        self.sat_max = float(online_bg_sat_max)
        self.val_min = float(online_bg_val_min)
        self.val_max = float(online_bg_val_max)
        self.hue_offset = float(online_bg_hue_offset)
        self.rng = random.Random()
        self.last_info = None

    def generate(self) -> Image.Image:
        base_size = self.base_size if self.use_base_size else self.size
        height = _diamond_square(base_size, self.rng)
        gamma = self.rng.uniform(self.gamma_min, self.gamma_max)
        height = np.power(height, gamma)
        hue = (height + (self.hue_offset / 256.0)) % 1.0
        sat = self.rng.uniform(self.sat_min, self.sat_max)
        val = self.rng.uniform(self.val_min, self.val_max)
        self.last_info = {
            "base_size": int(base_size),
            "gamma": float(gamma),
            "sat": float(sat),
            "val": float(val),
            "hue_offset": float(self.hue_offset),
            "blur_radius": float(self.blur_radius),
        }
        rgb = _hsv_to_rgb(hue, sat, val)
        img = (rgb * 255.0 + 0.5).astype(np.uint8)
        out = Image.fromarray(img, mode='RGB')
        if base_size != self.size:
            out = out.resize((self.size, self.size), Image.BILINEAR)
        if self.blur_radius and self.blur_radius > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return out

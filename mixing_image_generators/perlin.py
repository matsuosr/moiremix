"""On-the-fly Perlin/fBM generator for PixMix."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import BaseGenerator


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


@dataclass
class _PerlinConfig:
    mode: str
    size: int
    tileable: bool
    perlin_scale: float
    octave_min: int
    octave_max: int
    scale_min: float
    scale_max: float
    persistence_min: float
    persistence_max: float
    lacunarity_min: float
    lacunarity_max: float


class PerlinNoiseGenerator(BaseGenerator):
    """Fast Perlin / fBM noise generator."""

    def __init__(
        self,
        size: int = 224,
        online_perlin_mode: str = "fbm",
        online_perlin_tileable: bool = False,
        online_perlin_perlin_scale: float = 64.0,
        online_perlin_octaves_min: int = 4,
        online_perlin_octaves_max: int = 7,
        online_perlin_scale_min: float = 32.0,
        online_perlin_scale_max: float = 96.0,
        online_perlin_persistence_min: float = 0.45,
        online_perlin_persistence_max: float = 0.6,
        online_perlin_lacunarity_min: float = 1.8,
        online_perlin_lacunarity_max: float = 2.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = _PerlinConfig(
            mode=online_perlin_mode,
            size=int(size),
            tileable=bool(online_perlin_tileable),
            perlin_scale=float(max(1e-3, online_perlin_perlin_scale)),
            octave_min=max(1, int(online_perlin_octaves_min)),
            octave_max=max(1, int(online_perlin_octaves_max)),
            scale_min=float(max(1e-3, online_perlin_scale_min)),
            scale_max=float(max(1e-3, online_perlin_scale_max)),
            persistence_min=float(max(1e-4, online_perlin_persistence_min)),
            persistence_max=float(max(1e-4, online_perlin_persistence_max)),
            lacunarity_min=float(max(1e-4, online_perlin_lacunarity_min)),
            lacunarity_max=float(max(1e-4, online_perlin_lacunarity_max)),
        )
        if self.cfg.octave_max < self.cfg.octave_min:
            self.cfg.octave_max = self.cfg.octave_min
        if self.cfg.scale_max < self.cfg.scale_min:
            self.cfg.scale_max = self.cfg.scale_min
        if self.cfg.persistence_max < self.cfg.persistence_min:
            self.cfg.persistence_max = self.cfg.persistence_min
        if self.cfg.lacunarity_max < self.cfg.lacunarity_min:
            self.cfg.lacunarity_max = self.cfg.lacunarity_min
        self._rng = np.random.default_rng()
        self.last_info = None

    def _rand_grad_grid(self, gx: int, gy: int) -> np.ndarray:
        theta = self._rng.random((gy + 1, gx + 1)) * 2.0 * math.pi
        return np.stack((np.cos(theta), np.sin(theta)), axis=-1).astype(np.float32)

    def _perlin2d(self, width: int, height: int, scale: float) -> np.ndarray:
        scale = max(scale, 1e-3)
        gx = max(1, int(math.ceil(width / scale)))
        gy = max(1, int(math.ceil(height / scale)))
        grads = self._rand_grad_grid(gx, gy)

        y = np.linspace(0, gy, height, endpoint=False, dtype=np.float32)
        x = np.linspace(0, gx, width, endpoint=False, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="xy")

        x0 = np.floor(xx).astype(np.int32)
        y0 = np.floor(yy).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        xf = xx - x0
        yf = yy - y0

        if self.cfg.tileable:
            g00 = grads[y0 % gy, x0 % gx]
            g10 = grads[y0 % gy, x1 % gx]
            g01 = grads[y1 % gy, x0 % gx]
            g11 = grads[y1 % gy, x1 % gx]
        else:
            g00 = grads[y0, x0]
            g10 = grads[y0, x1]
            g01 = grads[y1, x0]
            g11 = grads[y1, x1]

        d00 = np.stack((xf, yf), axis=-1)
        d10 = np.stack((xf - 1.0, yf), axis=-1)
        d01 = np.stack((xf, yf - 1.0), axis=-1)
        d11 = np.stack((xf - 1.0, yf - 1.0), axis=-1)

        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)

        u = _fade(xf)
        v = _fade(yf)
        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        nxy = _lerp(nx0, nx1, v)

        nmin, nmax = float(nxy.min()), float(nxy.max())
        if nmax > nmin:
            nxy = (nxy - nmin) / (nmax - nmin)
        else:
            nxy = np.zeros_like(nxy)
        return nxy.astype(np.float32)

    def _fbm2d(
        self,
        width: int,
        height: int,
        octaves: int,
        base_scale: float,
        persistence: float,
        lacunarity: float,
    ) -> np.ndarray:
        total = np.zeros((height, width), dtype=np.float32)
        amp = 1.0
        freq_scale = base_scale
        amp_sum = 0.0
        lacunarity = max(lacunarity, 1e-4)
        persistence = max(persistence, 1e-4)
        for _ in range(octaves):
            total += amp * self._perlin2d(width, height, freq_scale)
            amp_sum += amp
            amp *= persistence
            freq_scale /= lacunarity
        if amp_sum > 0:
            total /= amp_sum
        return np.clip(total, 0.0, 1.0)

    def generate(self) -> Image.Image:
        size = self.cfg.size
        mode = self.cfg.mode
        if mode == "perlin":
            img = self._perlin2d(size, size, self.cfg.perlin_scale)
            self.last_info = {
                "mode": "perlin",
                "perlin_scale": float(self.cfg.perlin_scale),
            }
        else:
            octaves = int(
                self._rng.integers(self.cfg.octave_min, self.cfg.octave_max + 1)
            )
            base_scale = float(self._rng.uniform(self.cfg.scale_min, self.cfg.scale_max))
            persistence = float(
                self._rng.uniform(self.cfg.persistence_min, self.cfg.persistence_max)
            )
            lacunarity = float(
                self._rng.uniform(self.cfg.lacunarity_min, self.cfg.lacunarity_max)
            )
            img = self._fbm2d(size, size, octaves, base_scale, persistence, lacunarity)
            self.last_info = {
                "mode": str(mode),
                "octaves": int(octaves),
                "base_scale": float(base_scale),
                "persistence": float(persistence),
                "lacunarity": float(lacunarity),
            }

        rgb = np.repeat(img[..., None], 3, axis=-1)
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

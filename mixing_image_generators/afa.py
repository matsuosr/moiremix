"""On-the-fly AFA (CVPR 2024) Fourier basis generator (reference-aligned)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseGenerator


class AFAGenerator(BaseGenerator):
    """Generate AFA Fourier basis patterns for mixing.

    This implementation follows the frequency sampling and scaling scheme from
    https://github.com/nis-research/afa-augment (GeneralFourierOnline).
    """

    def __init__(
        self,
        size: int = 224,
        online_afa_min_str: Optional[float] = 0.0,
        online_afa_mean_str: Optional[float] = 10.0,
        online_afa_freq_cut: int = 1,
        online_afa_phase_cut: int = 1,
        online_afa_granularity: int = 448,
        online_afa_phase_min: float = 0.0,
        online_afa_phase_max: float = 1.0,
        online_afa_per_channel: bool = True,
        # Deprecated/legacy args (kept for backward compatibility)
        online_afa_lambda: Optional[float] = None,
        online_afa_f_min: Optional[float] = None,
        online_afa_f_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.per_channel = bool(online_afa_per_channel)

        mean_str = online_afa_mean_str
        if mean_str is None:
            mean_str = online_afa_lambda if online_afa_lambda is not None else 10.0
        self.mean_str = float(mean_str)
        self.min_str = float(online_afa_min_str if online_afa_min_str is not None else 0.0)

        self.freq_cut = max(1, int(online_afa_freq_cut))
        self.phase_cut = max(1, int(online_afa_phase_cut))
        self.granularity = max(1, int(online_afa_granularity))

        phase_min = float(online_afa_phase_min)
        phase_max = float(online_afa_phase_max)

        f_min = 1 if online_afa_f_min is None else int(round(online_afa_f_min))
        f_max = self.size if online_afa_f_max is None else int(round(online_afa_f_max))
        f_min = max(1, min(self.size, f_min))
        f_max = max(f_min, min(self.size, f_max))

        self.freqs = (np.arange(f_min, f_max + 1, dtype=np.float32) / float(self.size)).astype(
            np.float32
        )
        self.num_freqs = int(self.freqs.shape[0])
        self.phases = (-math.pi * np.linspace(phase_min, phase_max, num=self.granularity, dtype=np.float32)).astype(
            np.float32
        )
        self.num_phases = int(self.phases.shape[0])

        coord = np.linspace(-self.size / 2, self.size / 2, num=self.size, dtype=np.float32)
        self._x, self._y = np.meshgrid(coord, coord, indexing="ij")
        self.eps_scale = self.size / 32.0

        self.rng = np.random.default_rng()
        self.last_info = None

    def _sample_indices(self, channels: int) -> tuple[np.ndarray, np.ndarray]:
        if self.num_freqs <= 0 or self.num_phases <= 0:
            raise RuntimeError("AFA frequency/phase grids are empty. Check parameter ranges.")

        freq_idx = self.rng.integers(0, self.num_freqs, size=(channels, self.freq_cut, 1))
        phase_idx = self.rng.integers(0, self.num_phases, size=(channels, self.freq_cut, self.phase_cut))

        if not self.per_channel:
            freq_idx = np.repeat(freq_idx[:1], channels, axis=0)
            phase_idx = np.repeat(phase_idx[:1], channels, axis=0)

        return freq_idx, phase_idx

    def _sample_strengths(self, channels: int) -> np.ndarray:
        if self.mean_str <= 0:
            strengths = np.zeros((channels, self.freq_cut, self.phase_cut), dtype=np.float32)
        else:
            strengths = self.rng.exponential(
                scale=self.mean_str, size=(channels, self.freq_cut, self.phase_cut)
            ).astype(np.float32)
        strengths += self.min_str
        if not self.per_channel:
            strengths = np.repeat(strengths[:1], channels, axis=0)
        return strengths

    def generate(self) -> Image.Image:
        channels = 3
        freq_idx, phase_idx = self._sample_indices(channels)
        strengths = self._sample_strengths(channels)
        self.last_info = {
            "freq_cut": int(self.freq_cut),
            "phase_cut": int(self.phase_cut),
            "per_channel": bool(self.per_channel),
            "freq_indices": np.asarray(freq_idx).astype(int).tolist(),
            "phase_indices": np.asarray(phase_idx).astype(int).tolist(),
            "strengths_mean": float(np.asarray(strengths).mean()),
            "strengths_max": float(np.asarray(strengths).max()),
        }

        freqs = self.freqs[freq_idx]  # (C, F, 1)
        phases = self.phases[phase_idx]  # (C, F, P)

        freqs = freqs[..., None, None]
        phases = phases[..., None, None]
        waves = np.sin(
            2.0
            * math.pi
            * freqs
            * (self._x * np.cos(phases) + self._y * np.sin(phases))
            - (math.pi / 4.0)
        ).astype(np.float32)

        flat = waves.reshape(channels, self.freq_cut, self.phase_cut, -1)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        waves = (flat / norms).reshape(waves.shape) * self.eps_scale

        aug = (strengths[..., None, None] * waves).sum(axis=(1, 2))
        aug *= 1.0 / (self.freq_cut * self.phase_cut)

        base = 0.5
        img = np.clip(base + aug, 0.0, 1.0)
        rgb = np.transpose(img, (1, 2, 0))
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

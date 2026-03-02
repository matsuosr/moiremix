"""On-the-fly Fourier basis generator (NeurIPS 2019-style)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .base import BaseGenerator


class FourierBasis2019Generator(BaseGenerator):
    """Generate a real-valued Fourier basis image for mixing.

    This follows the NeurIPS 2019 definition where the DFT has two symmetric
    peaks (i,j) and (-i,-j), yielding a real-valued cosine basis. Each channel
    samples its own basis and sign.
    """

    def __init__(
        self,
        size: int = 224,
        online_fourier2019_mode: str = "uniform",
        online_fourier2019_r_min: float = 1.0,
        online_fourier2019_r_max: float = 50.0,
        online_fourier2019_amp: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.mode = online_fourier2019_mode
        self.r_min = float(online_fourier2019_r_min)
        self.r_max = float(online_fourier2019_r_max)
        self.amp = float(online_fourier2019_amp)
        self.rng = np.random.default_rng()
        self.last_info = None
        coord = np.linspace(0.0, 1.0, self.size, endpoint=False, dtype=np.float32)
        self.u, self.v = np.meshgrid(coord, coord, indexing="xy")
        self.k_max = max(1, self.size // 2)

    def _sample_ij(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "radial":
            r = self.rng.uniform(self.r_min, self.r_max, size=count)
            theta = self.rng.uniform(0.0, 2.0 * math.pi, size=count)
            i = np.rint(r * np.cos(theta)).astype(np.int32)
            j = np.rint(r * np.sin(theta)).astype(np.int32)
        else:
            i = self.rng.integers(-self.k_max, self.k_max + 1, size=count, dtype=np.int32)
            j = self.rng.integers(-self.k_max, self.k_max + 1, size=count, dtype=np.int32)

        i = np.clip(i, -self.k_max, self.k_max).astype(np.float32)
        j = np.clip(j, -self.k_max, self.k_max).astype(np.float32)
        return i, j

    def generate(self) -> Image.Image:
        i, j = self._sample_ij(3)
        signs = self.rng.choice([-1.0, 1.0], size=3).astype(np.float32)
        self.last_info = {
            "mode": str(self.mode),
            "i": [int(v) for v in i.tolist()],
            "j": [int(v) for v in j.tolist()],
            "signs": [int(v) for v in signs.tolist()],
            "amp": float(self.amp),
        }

        uu = self.u[None, :, :]
        vv = self.v[None, :, :]
        phase = 2.0 * math.pi * (i[:, None, None] * uu + j[:, None, None] * vv)
        basis = np.cos(phase).astype(np.float32)
        norm = np.linalg.norm(basis.reshape(3, -1), axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        basis = basis / norm[:, None, None]

        img = 0.5 + (signs[:, None, None] * self.amp) * basis
        img = np.clip(img, 0.0, 1.0)
        rgb = self._to_hwc_uint8(img)
        return Image.fromarray(rgb, mode="RGB")

    def _to_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        """Normalize any shape to HWC uint8 for visualization consistency."""
        img = np.asarray(img)
        img = np.squeeze(img)

        if img.ndim == 4:
            # Handle common 4D cases safely (e.g., B,3,H,W or B,H,W,3).
            if img.shape[-1] == 3 and img.shape[-2] == self.size and img.shape[-3] == self.size:
                # (B,H,W,3)
                img = img[0]
            elif img.shape[1] == 3 and img.shape[2] == self.size and img.shape[3] == self.size:
                # (B,3,H,W)
                img = img[0]
            elif img.shape[0] == 3 and img.shape[2] == self.size and img.shape[3] == self.size:
                # (3,B,H,W) or (3,3,H,W)
                img = img[:, 0, :, :]
            else:
                raise ValueError(f"Unexpected Fourier2019 4D image shape: {img.shape}")
            img = np.squeeze(img)

        if img.ndim == 2:
            rgb = np.repeat(img[..., None], 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[0] == 3 and img.shape[1] == self.size and img.shape[2] == self.size:
                rgb = np.transpose(img, (1, 2, 0))
            elif img.shape[2] == 3 and img.shape[0] == self.size and img.shape[1] == self.size:
                rgb = img
            elif img.shape[-1] == 1:
                rgb = np.repeat(img, 3, axis=-1)
            else:
                raise ValueError(f"Unexpected Fourier2019 3D image shape: {img.shape}")
        else:
            raise ValueError(f"Unexpected Fourier2019 image ndim: {img.ndim} shape: {img.shape}")

        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0 + 0.5).astype(np.uint8)

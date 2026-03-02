"""On-the-fly Dead Leaves generator for PixMix."""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .base import BaseGenerator


def _rand_color(rng: random.Random) -> Tuple[int, int, int]:
    h = rng.random()
    s = rng.uniform(0.4, 1.0)
    v = rng.uniform(0.6, 1.0)
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


def _regular_polygon(
    cx: float,
    cy: float,
    r: float,
    n_sides: int,
    angle_deg: float,
) -> List[Tuple[float, float]]:
    verts: List[Tuple[float, float]] = []
    base = math.radians(angle_deg)
    for k in range(n_sides):
        th = base + 2 * math.pi * k / n_sides
        verts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return verts


def _spectrum_texture(size: int, rng: random.Random) -> Image.Image:
    """Generate a colored 1/f-style texture."""
    H = W = size
    fy = np.fft.fftfreq(H).reshape(-1, 1)
    fx = np.fft.fftfreq(W).reshape(1, -1)

    def make_gray() -> np.ndarray:
        a = rng.uniform(0.8, 3.0)
        b = rng.uniform(0.8, 3.0)
        denom = np.power(np.abs(fx), a) + np.power(np.abs(fy), b)
        denom[0, 0] = 1.0
        mag = 1.0 / denom
        phase = np.exp(
            1j
            * 2
            * np.pi
            * np.random.default_rng(rng.randrange(1 << 31)).random((H, W))
        )
        img = np.fft.ifft2(mag * phase).real.astype(np.float32)
        lo, hi = np.percentile(img, (1.0, 99.0))
        img = (img - lo) / max(1e-6, hi - lo)
        return np.clip(img, 0.0, 1.0)

    channels = np.stack([make_gray(), make_gray(), make_gray()], axis=0)
    basis = np.array(
        [[rng.normalvariate(0, 1) for _ in range(3)] for _ in range(3)],
        dtype=np.float64,
    )
    q, _ = np.linalg.qr(basis)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    colored = (q.astype(np.float32) @ channels.reshape(3, -1)).reshape(3, H, W)
    colored = np.clip(colored, 0.0, 1.0)
    rgb = (colored * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")


class DeadLeavesGenerator(BaseGenerator):
    """Fast on-the-fly Dead Leaves generator."""

    def __init__(
        self,
        size: int = 224,
        online_deadleaves_variant: str = "shapes",
        online_deadleaves_shapes_min: int = 250,
        online_deadleaves_shapes_max: int = 400,
        online_deadleaves_radius_min: float = 4.0,
        online_deadleaves_radius_max: float = 40.0,
        online_deadleaves_bg: str = "uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.variant = online_deadleaves_variant
        self.shapes_min = max(1, int(online_deadleaves_shapes_min))
        self.shapes_max = max(self.shapes_min, int(online_deadleaves_shapes_max))
        self.radius_min = max(1.0, float(online_deadleaves_radius_min))
        self.radius_max = max(self.radius_min, float(online_deadleaves_radius_max))
        self.bg_mode = online_deadleaves_bg
        self.rng = random.Random()
        self.last_info = None

    def _background_color(self) -> Tuple[int, int, int]:
        if self.bg_mode == "black":
            return (0, 0, 0)
        if self.bg_mode == "white":
            return (255, 255, 255)
        # uniform pastel for default
        h = self.rng.random()
        s = self.rng.uniform(0.0, 0.25)
        v = self.rng.uniform(0.7, 1.0)
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

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        mask_draw: ImageDraw.ImageDraw | None,
        texture: Image.Image | None,
        shape_type: str,
        cx: float,
        cy: float,
        r: float,
        angle: float,
    ):
        color = _rand_color(self.rng)
        if shape_type == "circle":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=color)
            if mask_draw is not None:
                mask_draw.ellipse(bbox, fill=255)
            return

        if shape_type == "ellipse":
            aspect = self.rng.uniform(0.5, 2.0)
            rx = r
            ry = r * aspect
            polygon = _regular_polygon(cx, cy, r=max(rx, ry), n_sides=64, angle_deg=angle)
        elif shape_type == "triangle":
            polygon = _regular_polygon(cx, cy, r, n_sides=3, angle_deg=angle)
        elif shape_type == "pentagon":
            polygon = _regular_polygon(cx, cy, r, n_sides=5, angle_deg=angle)
        elif shape_type == "hexagon":
            polygon = _regular_polygon(cx, cy, r, n_sides=6, angle_deg=angle)
        else:
            polygon = _regular_polygon(cx, cy, r, n_sides=4, angle_deg=angle)

        draw.polygon(polygon, fill=color)
        if mask_draw is not None:
            mask_draw.polygon(polygon, fill=255)

        if texture is not None and mask_draw is not None:
            pass  # texture pasted separately

    def generate(self) -> Image.Image:
        size = self.size
        bg = self._background_color()
        canvas = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(canvas, "RGB")

        texture = None
        if self.variant == "textured":
            texture = _spectrum_texture(size, self.rng)

        num_shapes = self.rng.randint(self.shapes_min, self.shapes_max)
        self.last_info = {
            "variant": str(self.variant),
            "num_shapes": int(num_shapes),
            "radius_min": float(self.radius_min),
            "radius_max": float(self.radius_max),
            "bg_mode": str(self.bg_mode),
        }
        for _ in range(num_shapes):
            radius = self.rng.uniform(self.radius_min, self.radius_max)
            cx = self.rng.uniform(-radius, size + radius)
            cy = self.rng.uniform(-radius, size + radius)
            angle = self.rng.uniform(0.0, 360.0)

            if self.variant == "squares":
                shape_type = "square"
                angle = 0.0
            elif self.variant == "oriented":
                shape_type = "square"
            elif self.variant == "textured":
                # use curved quadrilateral mask
                mask = Image.new("L", (size, size), 0)
                mdraw = ImageDraw.Draw(mask, "L")
                poly = _regular_polygon(cx, cy, radius, n_sides=4, angle_deg=angle)
                mdraw.polygon(poly, fill=255)
                canvas.paste(texture, (0, 0), mask)
                continue
            else:
                shape_type = self.rng.choice(
                    ["circle", "triangle", "rect", "pentagon", "hexagon", "ellipse"]
                )

            self._draw_shape(
                draw=draw,
                mask_draw=None,
                texture=None,
                shape_type=shape_type,
                cx=cx,
                cy=cy,
                r=radius,
                angle=angle,
            )

        return canvas

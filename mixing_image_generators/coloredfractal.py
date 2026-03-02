"""On-the-fly Colored FractalDB generator for PixMix."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .base import BaseGenerator


@dataclass
class Affine:
    A: np.ndarray  # (2,2)
    b: np.ndarray  # (2,)
    p: float       # probability


def sample_contractive_A(rng: random.Random, scale_min: float, scale_max: float) -> np.ndarray:
    th = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    sx = rng.uniform(scale_min, scale_max)
    sy = rng.uniform(scale_min, scale_max)
    S = np.diag([sx, sy]).astype(np.float32)
    A = (R @ S @ R.T).astype(np.float32)
    _, svals, _ = np.linalg.svd(A)
    if svals[0] > scale_max:
        A = A * (scale_max / svals[0])
    return A


def sample_ifs(
    rng: random.Random,
    num_funcs: int,
    scale_min: float,
    scale_max: float,
    prob_mode: str = "det",
) -> List[Affine]:
    aff: List[Affine] = []
    for _ in range(num_funcs):
        A = sample_contractive_A(rng, scale_min, scale_max)
        r = rng.uniform(0.0, 1.0) ** 0.5
        ang = rng.uniform(0, 2 * math.pi)
        b = np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float32)
        aff.append(Affine(A=A, b=b, p=1.0))
    if prob_mode == "uniform":
        ps = np.ones((num_funcs,), dtype=np.float32)
    elif prob_mode == "det":
        ps = np.array([abs(np.linalg.det(a.A)) for a in aff], dtype=np.float32)
    else:
        ps = np.array([rng.uniform(0.5, 1.5) for _ in range(num_funcs)], dtype=np.float32)
    ps = ps / max(1e-8, ps.sum())
    for a, p in zip(aff, ps):
        a.p = float(p)
    return aff


def render_ifs_gray(
    ifs: List[Affine],
    size: int,
    iters: int,
    burnin: int,
    rng: random.Random,
    gamma: float = 0.7,
) -> Image.Image:
    x = np.zeros(2, dtype=np.float32)
    idxs = np.arange(len(ifs))
    P = np.array([a.p for a in ifs], dtype=np.float32)
    pts = np.zeros((iters + burnin, 2), dtype=np.float32)
    for i in range(iters + burnin):
        j = rng.choices(idxs, weights=P, k=1)[0]
        a = ifs[j]
        x = a.A @ x + a.b
        pts[i] = x
    cloud = pts[burnin:]
    mn = cloud.min(axis=0)
    mx = cloud.max(axis=0)
    span = (mx - mn)
    span[span < 1e-6] = 1.0
    uv = (cloud - mn) / span
    hw = size - 1
    ij = np.clip((uv * hw + 0.5).astype(np.int32), 0, size - 1)
    H = np.zeros((size, size), dtype=np.float32)
    np.add.at(H, (ij[:, 1], ij[:, 0]), 1.0)
    H = np.log1p(H)
    H = H / (H.max() + 1e-8)
    H = np.power(H, gamma)
    img = (H * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img, mode='L')


def bg_solid(size: int, rng: random.Random) -> Image.Image:
    h = rng.random()
    s = rng.uniform(0.2, 0.6)
    v = rng.uniform(0.8, 1.0)
    i = int(h * 6) % 6
    f = h * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    V = int(255 * v)
    if i == 0:
        rgb = (V, t, p)
    elif i == 1:
        rgb = (q, V, p)
    elif i == 2:
        rgb = (p, V, t)
    elif i == 3:
        rgb = (p, q, V)
    elif i == 4:
        rgb = (t, p, V)
    else:
        rgb = (V, p, q)
    return Image.new('RGB', (size, size), rgb)


def bg_linear(size: int, rng: random.Random) -> Image.Image:
    c1 = bg_solid(size, rng).getpixel((0, 0))
    c2 = bg_solid(size, rng).getpixel((0, 0))
    img = Image.new('RGB', (size, size), c1)
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        r = int((1 - t) * c1[0] + t * c2[0])
        g = int((1 - t) * c1[1] + t * c2[1])
        b = int((1 - t) * c1[2] + t * c2[2])
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def bg_softnoise(size: int, rng: random.Random) -> Image.Image:
    gsz = rng.choice([8, 12, 16, 20])
    grid_rng = np.random.default_rng(rng.randint(0, 1 << 31))
    grid = grid_rng.random((gsz, gsz)).astype(np.float32)
    grid = (grid * 255).astype(np.uint8)
    small = Image.fromarray(grid, mode='L')
    big = small.resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=1.1))
    base = bg_solid(size, rng).getpixel((0, 0))
    tweak = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in base)
    return ImageOps.colorize(big, black=base, white=tweak)


BG_CHOICES = [bg_solid, bg_linear, bg_softnoise]


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


def bg_diamond_square(
    size: int,
    rng: random.Random,
    gamma_min: float,
    gamma_max: float,
    sat_min: float,
    sat_max: float,
    val_min: float,
    val_max: float,
    hue_offset: float,
) -> Image.Image:
    height = _diamond_square(size, rng)
    gamma = rng.uniform(gamma_min, gamma_max)
    height = np.power(height, gamma)
    hue = (height + hue_offset) % 1.0
    sat = rng.uniform(sat_min, sat_max)
    val = rng.uniform(val_min, val_max)
    rgb = _hsv_to_rgb(hue, sat, val)
    img = (rgb * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img, mode='RGB')


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
    elif i == 1:
        return (q, V, p)
    elif i == 2:
        return (p, V, t)
    elif i == 3:
        return (p, q, V)
    elif i == 4:
        return (t, p, V)
    else:
        return (V, p, q)


def colorize_with_alpha(gray: Image.Image, rng: random.Random) -> Image.Image:
    col1 = random_color(rng)
    col2 = random_color(rng)
    if rng.random() < 0.5:
        col1, col2 = col2, col1
    rgb = ImageOps.colorize(gray, black=col1, white=col2)
    a = ImageOps.autocontrast(gray)
    a = np.asarray(a, dtype=np.float32) / 255.0
    a = np.power(a, 1.15)
    a = (a * 255 + 0.5).astype(np.uint8)
    rgba = rgb.convert('RGBA')
    rgba.putalpha(Image.fromarray(a, mode='L'))
    return rgba


def colorize_with_hsv(
    gray: Image.Image,
    rng: random.Random,
    sat_min: float,
    sat_max: float,
    val_min: float,
    val_max: float,
    hue_offset: float,
    alpha_pow: float,
) -> Image.Image:
    g = np.asarray(gray, dtype=np.float32) / 255.0
    hue = (g + hue_offset) % 1.0
    sat = rng.uniform(sat_min, sat_max)
    val = rng.uniform(val_min, val_max)
    rgb = _hsv_to_rgb(hue, sat, val)
    a = np.power(g, alpha_pow)
    rgba = (rgb * 255.0 + 0.5).astype(np.uint8)
    alpha = (a * 255.0 + 0.5).astype(np.uint8)
    out = Image.fromarray(rgba, mode='RGB').convert('RGBA')
    out.putalpha(Image.fromarray(alpha, mode='L'))
    return out


class FractalGenerator(BaseGenerator):
    def __init__(
        self,
        size: int = 224,
        online_fractal_iters: int = 40000,
        online_fractal_instances_min: int = 1,
        online_fractal_instances_max: int = 3,
        online_fractal_scale_min: float = 0.4,
        online_fractal_scale_max: float = 0.85,
        online_fractal_bg_mode: str = "original",
        online_fractal_color_mode: str = "original",
        online_fractal_prob_mode: str = "det",
        online_fractal_num_funcs_mode: str = "original",
        online_fractal_gamma_min: float = 0.4,
        online_fractal_gamma_max: float = 0.8,
        online_fractal_sat_min: float = 0.3,
        online_fractal_sat_max: float = 1.0,
        online_fractal_val_min: float = 0.5,
        online_fractal_val_max: float = 1.0,
        online_fractal_hue_offset: Optional[float] = 0.0,
        online_fractal_alpha_pow: float = 1.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = int(size)
        self.iters = int(online_fractal_iters)
        self.instances_min = max(1, int(online_fractal_instances_min))
        self.instances_max = max(self.instances_min, int(online_fractal_instances_max))
        self.scale_min = float(online_fractal_scale_min)
        self.scale_max = float(online_fractal_scale_max)
        self.bg_mode = str(online_fractal_bg_mode).lower()
        self.color_mode = str(online_fractal_color_mode).lower()
        self.prob_mode = str(online_fractal_prob_mode).lower()
        self.num_funcs_mode = str(online_fractal_num_funcs_mode).lower()
        self.gamma_min = float(online_fractal_gamma_min)
        self.gamma_max = float(online_fractal_gamma_max)
        self.sat_min = float(online_fractal_sat_min)
        self.sat_max = float(online_fractal_sat_max)
        self.val_min = float(online_fractal_val_min)
        self.val_max = float(online_fractal_val_max)
        self.hue_offset = 0.0 if online_fractal_hue_offset is None else float(online_fractal_hue_offset)
        self.alpha_pow = float(online_fractal_alpha_pow)
        self.rng = random.Random()
        self.last_info = None

    def generate(self) -> Image.Image:
        if self.bg_mode == "legacy":
            base_bg = self.rng.choice(BG_CHOICES)(self.size, self.rng)
        else:
            base_bg = bg_diamond_square(
                self.size,
                self.rng,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                sat_min=self.sat_min,
                sat_max=self.sat_max,
                val_min=self.val_min,
                val_max=self.val_max,
                hue_offset=self.hue_offset,
            )
        canvas = base_bg.convert('RGBA')
        layers = self.rng.randint(self.instances_min, self.instances_max)
        layer_num_funcs = []
        for _ in range(layers):
            if self.num_funcs_mode == "legacy":
                k = self.rng.randint(3, 6)
            else:
                k = self.rng.choice([2, 3, 4])
            layer_num_funcs.append(int(k))
            ifs = sample_ifs(self.rng, k, self.scale_min, self.scale_max, prob_mode=self.prob_mode)
            gray = render_ifs_gray(
                ifs,
                size=self.size,
                iters=self.iters,
                burnin=max(64, self.iters // 6),
                rng=self.rng,
            )
            if self.color_mode == "legacy":
                colored = colorize_with_alpha(gray, self.rng)
            else:
                colored = colorize_with_hsv(
                    gray,
                    self.rng,
                    sat_min=self.sat_min,
                    sat_max=self.sat_max,
                    val_min=self.val_min,
                    val_max=self.val_max,
                    hue_offset=self.hue_offset,
                    alpha_pow=self.alpha_pow,
                )
            canvas = Image.alpha_composite(canvas, colored)
        self.last_info = {
            "layers": int(layers),
            "num_funcs": layer_num_funcs,
            "scale_min": float(self.scale_min),
            "scale_max": float(self.scale_max),
            "iters": int(self.iters),
            "bg_mode": str(self.bg_mode),
            "color_mode": str(self.color_mode),
            "prob_mode": str(self.prob_mode),
        }
        return canvas.convert('RGB')

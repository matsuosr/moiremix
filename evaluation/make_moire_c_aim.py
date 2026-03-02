#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Moire-C (AIM 2019 LCDMoire-inspired) corruption for ImageNet-C layout.
"""

import argparse
import hashlib
import os
import shutil
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


SEVERITY_TABLE = {
    1: dict(proj_max=0.02, blur_sigma=0.2, raw_noise_sigma=0.002, jpeg_q=95, mix_alpha=0.20),
    2: dict(proj_max=0.05, blur_sigma=0.4, raw_noise_sigma=0.004, jpeg_q=85, mix_alpha=0.35),
    3: dict(proj_max=0.10, blur_sigma=0.7, raw_noise_sigma=0.007, jpeg_q=75, mix_alpha=0.50),
    4: dict(proj_max=0.15, blur_sigma=1.0, raw_noise_sigma=0.012, jpeg_q=60, mix_alpha=0.65),
    5: dict(proj_max=0.20, blur_sigma=1.4, raw_noise_sigma=0.020, jpeg_q=45, mix_alpha=0.80),
}

LUMA_MATCH_ENABLED = True
LUMA_MIN_GAIN = 0.25
LUMA_MAX_GAIN = 4.0
DUMP_CLEAN = False
CLEAN_MODE = "symlink"
RGB_MEAN_MATCH_ENABLED = True
RGB_MIN_GAIN = 0.5
RGB_MAX_GAIN = 2.0
DENOISE_ENABLED = False
DOWNSAMPLE_INTERP = cv2.INTER_LINEAR
CHROMA_MATCH_MODE = "mean"


def _seed_from(global_seed: int, severity: int, wnid: str, filename: str) -> int:
    payload = f"{global_seed}|{severity}|{wnid}|{filename}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _lcd_mosaic(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    out = np.zeros((h * 3, w * 3, 3), dtype=np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    out[1::3, 0::3, 0] = r
    out[1::3, 1::3, 1] = g
    out[1::3, 2::3, 2] = b
    out[2::3, 0::3, 0] = r
    out[2::3, 1::3, 1] = g
    out[2::3, 2::3, 2] = b
    # Keep high-resolution subpixel pattern; camera sampling happens later.
    return out


def _random_homography(img: np.ndarray, proj_max: float, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = img.shape
    # img is 3x upsampled LCD mosaic; match offset scale to original resolution.
    max_offset = proj_max * (min(h, w) / 3.0)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = src.copy()
    for i in range(4):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        # Uniform-in-area sampling inside a circle
        radius = np.sqrt(rng.uniform(0.0, 1.0)) * max_offset
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        dst[i, 0] += dx
        dst[i, 1] += dy
    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return warped


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = int(max(3, round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def _bayer_mosaic_rgb(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    raw = np.zeros((h, w), dtype=np.float32)
    raw[0::2, 0::2] = img[0::2, 0::2, 0]  # R
    raw[0::2, 1::2] = img[0::2, 1::2, 1]  # G
    raw[1::2, 0::2] = img[1::2, 0::2, 1]  # G
    raw[1::2, 1::2] = img[1::2, 1::2, 2]  # B
    return raw


def _demosaic_and_denoise(raw: np.ndarray) -> np.ndarray:
    raw_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(raw_u8, cv2.COLOR_BayerRG2BGR)
    if DENOISE_ENABLED:
        return cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
    return bgr


def _mean_luma(rgb01: np.ndarray) -> float:
    r = rgb01[:, :, 0]
    g = rgb01[:, :, 1]
    b = rgb01[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return float(y.mean())


def _luma_match_bgr(out_bgr_u8: np.ndarray, in_rgb01: np.ndarray, min_gain: float, max_gain: float) -> np.ndarray:
    out_rgb = cv2.cvtColor(out_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    yin = _mean_luma(in_rgb01)
    yout = _mean_luma(out_rgb)
    gain = yin / (yout + 1e-6)
    gain = float(np.clip(gain, min_gain, max_gain))
    out_rgb = np.clip(out_rgb * gain, 0.0, 1.0)
    out_bgr = cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr


def _chroma_match_bgr(out_bgr_u8: np.ndarray, in_rgb01: np.ndarray) -> np.ndarray:
    in_bgr = cv2.cvtColor((in_rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    in_lab = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2LAB)
    out_lab = cv2.cvtColor(out_bgr_u8, cv2.COLOR_BGR2LAB)
    a_shift = float(in_lab[:, :, 1].mean() - out_lab[:, :, 1].mean())
    b_shift = float(in_lab[:, :, 2].mean() - out_lab[:, :, 2].mean())
    out_lab = out_lab.astype(np.float32)
    out_lab[:, :, 1] = np.clip(out_lab[:, :, 1] + a_shift, 0, 255)
    out_lab[:, :, 2] = np.clip(out_lab[:, :, 2] + b_shift, 0, 255)
    out_lab = out_lab.astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def _rgb_mean_match_bgr(out_bgr_u8: np.ndarray, in_rgb01: np.ndarray, min_gain: float, max_gain: float) -> np.ndarray:
    out_rgb = cv2.cvtColor(out_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    in_m = in_rgb01.reshape(-1, 3).mean(axis=0)
    out_m = out_rgb.reshape(-1, 3).mean(axis=0)
    gains = in_m / (out_m + 1e-6)
    gains = np.clip(gains, min_gain, max_gain).astype(np.float32)
    out_rgb = np.clip(out_rgb * gains[None, None, :], 0.0, 1.0)
    out_bgr = cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr


def _apply_pipeline(img_rgb: np.ndarray, params: Dict[str, float], rng: np.random.Generator) -> np.ndarray:
    # 1) LCD subpixel mosaic (3H x 3W)
    lcd_hi = _lcd_mosaic(img_rgb)
    # 2) random homography + Gaussian blur on high-res "display"
    warped_hi = _random_homography(lcd_hi, params["proj_max"], rng)
    blurred_hi = _gaussian_blur(warped_hi, params["blur_sigma"])
    # 3) camera sampling back to sensor resolution (H x W)
    h, w, _ = img_rgb.shape
    cam = cv2.resize(blurred_hi, (w, h), interpolation=DOWNSAMPLE_INTERP)
    # 4) Bayer CFA sampling + noise
    raw = _bayer_mosaic_rgb(cam)
    raw += rng.normal(0.0, params["raw_noise_sigma"], size=raw.shape).astype(np.float32)
    raw = np.clip(raw, 0.0, 1.0)
    bgr = _demosaic_and_denoise(raw)
    return bgr


def _dump_clean(src_path: str, out_dir: str, wnid: str, filename: str) -> None:
    dst_path = os.path.join(out_dir, "_clean", wnid, filename)
    if os.path.exists(dst_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if CLEAN_MODE == "symlink":
        os.symlink(src_path, dst_path)
    elif CLEAN_MODE == "hardlink":
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def _process_one(task: Tuple[str, str, str, int, int, str]) -> str:
    src_path, wnid, filename, severity, global_seed, out_dir = task
    out_path = os.path.join(out_dir, str(severity), wnid, filename)
    if os.path.isfile(out_path):
        return "skipped"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if DUMP_CLEAN and severity == 1:
        _dump_clean(src_path, out_dir, wnid, filename)

    seed = _seed_from(global_seed, severity, wnid, filename)
    rng = np.random.default_rng(seed)

    bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return "failed_read"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    params = SEVERITY_TABLE[severity]
    out_bgr = _apply_pipeline(rgb, params, rng)
    if RGB_MEAN_MATCH_ENABLED:
        out_bgr = _rgb_mean_match_bgr(out_bgr, rgb, RGB_MIN_GAIN, RGB_MAX_GAIN)
    if CHROMA_MATCH_MODE == "mean":
        out_bgr = _chroma_match_bgr(out_bgr, rgb)
    if LUMA_MATCH_ENABLED:
        out_bgr = _luma_match_bgr(out_bgr, rgb, LUMA_MIN_GAIN, LUMA_MAX_GAIN)
    clean_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    alpha = float(params.get("mix_alpha", 1.0))
    out_bgr = np.clip(
        (1.0 - alpha) * clean_bgr.astype(np.float32) + alpha * out_bgr.astype(np.float32),
        0.0,
        255.0,
    ).astype(np.uint8)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(params["jpeg_q"])]
    ok, buf = cv2.imencode(".jpg", out_bgr, encode_params)
    if not ok:
        return "failed_encode"
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return "ok"


def _collect_images(imagenet_val: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for wnid in sorted(os.listdir(imagenet_val)):
        wnid_dir = os.path.join(imagenet_val, wnid)
        if not os.path.isdir(wnid_dir):
            continue
        for fname in sorted(os.listdir(wnid_dir)):
            src = os.path.join(wnid_dir, fname)
            if not os.path.isfile(src):
                continue
            items.append((src, wnid, fname))
    return items


def _iter_tasks(
    images: Iterable[Tuple[str, str, str]],
    out_dir: str,
    global_seed: int,
    severities: Iterable[int],
) -> Iterable[Tuple[str, str, str, int, int, str]]:
    for src, wnid, fname in images:
        for severity in severities:
            yield (src, wnid, fname, severity, global_seed, out_dir)


def _check_missing(images: List[Tuple[str, str, str]], out_dir: str) -> int:
    missing = 0
    for _, wnid, fname in images:
        for severity in range(1, 6):
            out_path = os.path.join(out_dir, str(severity), wnid, fname)
            if not os.path.isfile(out_path):
                missing += 1
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Moire-C (AIM LCDMoire) for ImageNet-C layout.")
    parser.add_argument("--imagenet-val", required=True, help="Path to ImageNet-1K val directory")
    parser.add_argument("--out", required=True, help="Output directory (e.g., /disk4/datasets/ImageNet-C/moire_aim)")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of source images (0 = all)")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=1, help="Global seed for deterministic generation")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5",
                        help="Comma-separated severities to generate (e.g., 4 or 1,3,5)")
    parser.add_argument("--check-results", action="store_true", help="Check for missing outputs after generation")
    parser.add_argument("--no-luma-match", action="store_true", help="Disable luma matching")
    parser.add_argument("--luma-min-gain", type=float, default=0.25, help="Minimum luma gain")
    parser.add_argument("--luma-max-gain", type=float, default=4.0, help="Maximum luma gain")
    parser.add_argument("--no-rgb-mean-match", action="store_true", help="Disable RGB mean matching")
    parser.add_argument("--rgb-min-gain", type=float, default=0.5, help="Minimum RGB mean gain")
    parser.add_argument("--rgb-max-gain", type=float, default=2.0, help="Maximum RGB mean gain")
    parser.add_argument("--chroma-match", choices=["none", "mean"], default="mean",
                        help="Chroma match mode in Lab space")
    dump_group = parser.add_mutually_exclusive_group()
    dump_group.add_argument("--dump-clean", dest="dump_clean", action="store_true",
                            help="Dump clean images under _clean")
    dump_group.add_argument("--no-dump-clean", dest="dump_clean", action="store_false",
                            help="Disable clean image dumping")
    parser.set_defaults(dump_clean=None)
    parser.add_argument("--clean-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                        help="Clean image output mode")
    parser.add_argument("--downsample-interp", choices=["area", "linear", "nearest"], default="linear",
                        help="Downsample interpolation for LCD->sensor")
    denoise_group = parser.add_mutually_exclusive_group()
    denoise_group.add_argument("--denoise", dest="denoise", action="store_true",
                               help="Enable denoising after demosaic")
    denoise_group.add_argument("--no-denoise", dest="denoise", action="store_false",
                               help="Disable denoising after demosaic")
    parser.set_defaults(denoise=False)
    args = parser.parse_args()

    if not os.path.isdir(args.imagenet_val):
        raise FileNotFoundError(args.imagenet_val)
    os.makedirs(args.out, exist_ok=True)
    global LUMA_MATCH_ENABLED, LUMA_MIN_GAIN, LUMA_MAX_GAIN, DUMP_CLEAN, CLEAN_MODE
    global RGB_MEAN_MATCH_ENABLED, RGB_MIN_GAIN, RGB_MAX_GAIN, DENOISE_ENABLED, DOWNSAMPLE_INTERP
    global CHROMA_MATCH_MODE
    LUMA_MATCH_ENABLED = not args.no_luma_match
    LUMA_MIN_GAIN = float(args.luma_min_gain)
    LUMA_MAX_GAIN = float(args.luma_max_gain)
    RGB_MEAN_MATCH_ENABLED = not args.no_rgb_mean_match
    RGB_MIN_GAIN = float(args.rgb_min_gain)
    RGB_MAX_GAIN = float(args.rgb_max_gain)
    CHROMA_MATCH_MODE = str(args.chroma_match)
    if args.dump_clean is None:
        DUMP_CLEAN = args.max_images > 0
    else:
        DUMP_CLEAN = bool(args.dump_clean)
    CLEAN_MODE = str(args.clean_mode)
    DENOISE_ENABLED = bool(args.denoise)
    if args.downsample_interp == "area":
        DOWNSAMPLE_INTERP = cv2.INTER_AREA
    elif args.downsample_interp == "nearest":
        DOWNSAMPLE_INTERP = cv2.INTER_NEAREST
    else:
        DOWNSAMPLE_INTERP = cv2.INTER_LINEAR

    images = _collect_images(args.imagenet_val)
    if args.max_images > 0:
        images = images[: args.max_images]

    severities = []
    for part in args.severities.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            severities.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid severity: {part}")
    if not severities:
        raise ValueError("No valid severities specified")
    for sev in severities:
        if sev not in SEVERITY_TABLE:
            raise ValueError(f"Severity out of range (1..5): {sev}")
    tasks = list(_iter_tasks(images, args.out, args.seed, severities))
    total = len(tasks)
    if total == 0:
        print("No input images found.")
        return

    stats = {"ok": 0, "skipped": 0, "failed_read": 0, "failed_encode": 0}
    with Pool(processes=args.num_workers) as pool:
        for idx, status in enumerate(pool.imap_unordered(_process_one, tasks), 1):
            if status in stats:
                stats[status] += 1
            if idx % 200 == 0 or idx == total:
                print(f"[{idx}/{total}] ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed_read'] + stats['failed_encode']}")

    print("Done.", stats)
    if args.check_results:
        missing = _check_missing(images, args.out)
        print(f"[check] missing_files={missing}")


if __name__ == "__main__":
    main()

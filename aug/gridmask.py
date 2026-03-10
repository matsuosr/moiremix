import math
import numpy as np
from PIL import Image


def _rng_float(rng):
    if hasattr(rng, "random"):
        return float(rng.random())
    if hasattr(rng, "rand"):
        return float(rng.rand())
    return float(rng.random())


def _rng_int(rng, low, high):
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    if hasattr(rng, "randint"):
        return int(rng.randint(low, high))
    return int(rng.randrange(low, high))


class GridMask:
    """GridMask augmentation (paper/official implementation aligned)."""

    def __init__(self, d_min=96, d_max=224, rotate=360, ratio=0.6, mode=1, prob=0.8, fill=0.0):
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.rotate = int(rotate)
        self.ratio = float(ratio)
        self.mode = int(mode)
        self.st_prob = float(prob)
        self.prob = float(prob)
        self.fill = 0.0 if fill is None else float(fill)

    def set_prob(self, epoch, max_epoch):
        max_epoch = max(1.0, float(max_epoch))
        self.prob = self.st_prob * min(1.0, float(epoch) / max_epoch)

    def _build_mask(self, h, w, rng, return_mask=False):
        info = {}
        hh = math.ceil(math.sqrt(h * h + w * w))
        d = _rng_int(rng, self.d_min, self.d_max)
        l = max(1, int(math.ceil(d * self.ratio)))
        st_h = _rng_int(rng, 0, d)
        st_w = _rng_int(rng, 0, d)

        mask = np.ones((hh, hh), np.float32)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0

        angle = _rng_int(rng, 0, max(1, self.rotate)) if self.rotate > 0 else 0
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        if self.rotate > 0:
            mask_img = mask_img.rotate(angle, resample=Image.NEAREST)
        mask = np.asarray(mask_img)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
        mask = (mask.astype(np.float32) / 255.0)
        if self.mode == 1:
            mask = 1.0 - mask

        info["mask_mean"] = float(mask.mean())

        keep_ratio = 2 * self.ratio - self.ratio * self.ratio if self.mode == 1 else (1 - self.ratio) ** 2
        info.update({
            "d": d,
            "l": l,
            "delta_x": st_w,
            "delta_y": st_h,
            "rotation_angle": angle,
            "p_current": self.prob,
            "r": self.ratio,
            "mode": self.mode,
            "keep_ratio_est": keep_ratio,
        })
        if return_mask:
            return mask, info
        return None, info

    def apply(self, img, rng=None, return_info=False):
        rng = np.random if rng is None else rng
        if _rng_float(rng) > self.prob:
            if return_info:
                return img, {"skipped": True, "p_current": self.prob}
            return img

        if isinstance(img, Image.Image):
            img_np = np.array(img).astype(np.float32) / 255.0
        else:
            img_np = np.array(img).astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        h, w = img_np.shape[:2]
        mask, info = self._build_mask(h, w, rng, return_mask=True)
        mask = mask[..., None]
        img_np = img_np * mask + self.fill * (1.0 - mask)
        out = np.clip(img_np, 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
        out = Image.fromarray(out, mode="RGB")
        if return_info:
            info["skipped"] = False
            info["mask_array"] = mask[..., 0]
            return out, info
        return out

    def __call__(self, img):
        return self.apply(img)


if __name__ == "__main__":
    gm = GridMask()
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
    _ = gm.apply(dummy, rng=np.random.default_rng(0))
    _ = gm.apply(dummy, rng=np.random.RandomState(0))

import os
import random

import numpy as np
import torch
import torch.utils.data

from . import ipmix_ops as ops


def _augment_input(image, aug_severity, all_ops):
    aug_list = ops.augmentations_all if all_ops else ops.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


def ipmix(image, mixing_pic, preprocess, k, t, beta, aug_severity, all_ops):
    # Ported from hzlsaber/IPMix imagenet.py (ipmix function).
    mixings = ops.mixings
    patch_sizes = [4, 8, 16, 32, 64, 224]
    mixing_ops = ['Img', 'P']
    patch_mixing = ops.patch_mixing

    tensorize = preprocess['tensorize']
    normalize = preprocess['normalize']

    def preprocess_fn(im):
        return normalize(tensorize(im))

    ws = np.float32(np.random.dirichlet([1] * k))
    m = np.float32(np.random.beta(1, 1))
    mix = torch.zeros_like(preprocess_fn(image))

    for i in range(k):
        mixed = image.copy()
        mixing_ways = random.choice(mixing_ops)
        if mixing_ways == 'P':
            for _ in range(np.random.randint(t + 1)):
                patch_size = random.choice(patch_sizes)
                mix_op = random.choice(mixings)
                if random.random() > 0.5:
                    mixed = patch_mixing(mixed, mixing_pic, patch_size, mix_op, beta)
                else:
                    mixed_copy = _augment_input(image, aug_severity, all_ops)
                    mixed = patch_mixing(mixed, mixed_copy, patch_size, mix_op, beta)
        else:
            for _ in range(np.random.randint(t + 1)):
                mixed = _augment_input(mixed, aug_severity, all_ops)
        mix += ws[i] * preprocess_fn(mixed)

    mix_result = (1 - m) * preprocess_fn(image) + m * mix
    return mix_result


def ipmix_with_overrides(
    image,
    mixing_pic,
    preprocess,
    k,
    t,
    beta,
    aug_severity,
    all_ops,
    m_min=None,
    force_nonzero_steps=False,
    patch_sizes=None,
):
    mixings = ops.mixings
    default_patch_sizes = [4, 8, 16, 32, 64, 224]
    patch_sizes = patch_sizes if patch_sizes else default_patch_sizes
    mixing_ops = ['Img', 'P']
    patch_mixing = ops.patch_mixing

    tensorize = preprocess['tensorize']
    normalize = preprocess['normalize']

    def preprocess_fn(im):
        return normalize(tensorize(im))

    ws = np.float32(np.random.dirichlet([1] * k))
    m = np.float32(np.random.beta(1, 1))
    if m_min is not None:
        m = np.float32(max(float(m), float(m_min)))
    mix = torch.zeros_like(preprocess_fn(image))

    for i in range(k):
        mixed = image.copy()
        mixing_ways = random.choice(mixing_ops)
        if mixing_ways == 'P':
            steps = np.random.randint(t + 1)
            if force_nonzero_steps and steps == 0:
                steps = 1
            for _ in range(steps):
                patch_size = random.choice(patch_sizes)
                mix_op = random.choice(mixings)
                if random.random() > 0.5:
                    mixed = patch_mixing(mixed, mixing_pic, patch_size, mix_op, beta)
                else:
                    mixed_copy = _augment_input(image, aug_severity, all_ops)
                    mixed = patch_mixing(mixed, mixed_copy, patch_size, mix_op, beta)
        else:
            steps = np.random.randint(t + 1)
            if force_nonzero_steps and steps == 0:
                steps = 1
            for _ in range(steps):
                mixed = _augment_input(mixed, aug_severity, all_ops)
        mix += ws[i] * preprocess_fn(mixed)

    mix_result = (1 - m) * preprocess_fn(image) + m * mix
    return mix_result


class IPMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        mixing_set,
        preprocess,
        k=3,
        t=3,
        beta=4,
        aug_severity=1,
        all_ops=False,
    ):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.k = k
        self.t = t
        self.beta = beta
        self.aug_severity = aug_severity
        self.all_ops = all_ops

        try:
            self.num_samples = len(self.dataset)
        except Exception:
            self.num_samples = len(self.mixing_set)

        if self.num_samples == 0:
            raise ValueError("IPMixDataset: base dataset is empty")
        if len(self.mixing_set) == 0:
            raise ValueError("IPMixDataset: mixing_set is empty")

        print(
            f"[IPMixDataset] len(dataset)={len(self.dataset)} "
            f"len(mixing_set)={len(self.mixing_set)}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        x, y = self.dataset[i]
        rnd_idx = random.randrange(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        out = ipmix(
            x, mixing_pic, self.preprocess,
            k=self.k, t=self.t, beta=self.beta,
            aug_severity=self.aug_severity, all_ops=self.all_ops
        )
        if os.getenv("IPMIX_DEBUG", "0") == "1" and not hasattr(self, "_debug_printed"):
            print(f"[IPMix] k={self.k} t={self.t} beta={self.beta}")
            self._debug_printed = True
        return out, y

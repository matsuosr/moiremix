import random
import numpy as np
import torch
import torch.utils.data


class DiffuseMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        mixing_set,
        preprocess,
        fractal_set=None,
        alpha=0.5,
        beta=0.5,
        concat_prob=0.5,
        fractal_lambda=None,
    ):
        # mixing_set is expected to be a pre-generated DiffuseMix-style image bank
        # when full diffusion inference is not run inside this repo.
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.fractal_set = fractal_set
        self.preprocess = preprocess
        self.alpha = alpha
        self.beta = beta
        self.concat_prob = concat_prob
        self.fractal_lambda = fractal_lambda

        self.num_samples = len(self.dataset)

        if self.num_samples == 0:
            raise ValueError("DiffuseMixDataset: base dataset is empty")
        if len(self.mixing_set) == 0:
            raise ValueError("DiffuseMixDataset: mixing_set is empty")
        if self.fractal_set is not None and len(self.fractal_set) == 0:
            raise ValueError("DiffuseMixDataset: fractal_set is empty")

        print(
            f"[DiffuseMixDataset] len(dataset)={len(self.dataset)} "
            f"len(mixing_set)={len(self.mixing_set)} "
            f"len(fractal_set)={len(self.fractal_set) if self.fractal_set is not None else 0}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        mix_idx = random.randrange(len(self.mixing_set))
        mix_img, _ = self.mixing_set[mix_idx]

        fractal_img = None
        if self.fractal_set is not None:
            fractal_idx = random.randrange(len(self.fractal_set))
            fractal_img, _ = self.fractal_set[fractal_idx]

        tensorize = self.preprocess['tensorize']
        normalize = self.preprocess['normalize']

        img_t = tensorize(img)
        mix_t = tensorize(mix_img)

        if fractal_img is not None and random.random() < self.concat_prob:
            fractal_t = tensorize(fractal_img)
            if self.fractal_lambda is None:
                lam_f = 0.5
            else:
                lam_f = float(self.fractal_lambda)
            mix_t = torch.clamp((1.0 - lam_f) * mix_t + lam_f * fractal_t, 0.0, 1.0)

        lam = np.random.beta(self.alpha, self.beta) if self.alpha > 0 and self.beta > 0 else 0.5
        mixed = torch.clamp(lam * img_t + (1.0 - lam) * mix_t, 0.0, 1.0)
        mixed = normalize(mixed)
        return mixed, target

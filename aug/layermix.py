import random
import numpy as np
import torch
import torch.utils.data


class LayerMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        mixing_set,
        preprocess,
        depth=3,
        width=3,
        magnitude=3,
        blending=3.0,
        use_all_ops=False,
    ):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.depth = depth
        self.width = width
        self.magnitude = magnitude
        self.blending = blending
        self.use_all_ops = use_all_ops

        self.num_samples = len(self.dataset)

        if self.num_samples == 0:
            raise ValueError("LayerMixDataset: base dataset is empty")
        if len(self.mixing_set) == 0:
            raise ValueError("LayerMixDataset: mixing_set is empty")

        print(
            f"[LayerMixDataset] len(dataset)={len(self.dataset)} "
            f"len(mixing_set)={len(self.mixing_set)}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        mix_idx = random.randrange(len(self.mixing_set))
        mix_img, _ = self.mixing_set[mix_idx]

        tensorize = self.preprocess['tensorize']
        normalize = self.preprocess['normalize']

        img_t = tensorize(img)
        mix_t = tensorize(mix_img)

        lam = np.random.beta(self.blending, self.blending) if self.blending > 0 else 0.5
        mixed = torch.clamp(lam * img_t + (1.0 - lam) * mix_t, 0.0, 1.0)
        mixed = normalize(mixed)
        return mixed, target

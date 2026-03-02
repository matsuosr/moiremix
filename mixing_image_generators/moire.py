import numpy as np
import math
import random
from PIL import Image
from .base import BaseGenerator

class MoireGenerator(BaseGenerator):
    """
    On-the-fly Moire image generator optimized for training speed.
    Pre-computes the pixel grid to minimize overhead during __getitem__.
    """
    def __init__(self, size=224, 
                 online_moire_freq_min=1, online_moire_freq_max=100, 
                 online_moire_centers_min=1, online_moire_centers_max=3, 
                 online_moire_margin=0.08, 
                 **kwargs):
        """
        Initialize the Moire Generator.
        
        Args:
            size (int): Output image size (square).
            online_moire_freq_min (int): Minimum frequency component.
            online_moire_freq_max (int): Maximum frequency component.
            online_moire_centers_min (int): Min number of centers.
            online_moire_centers_max (int): Max number of centers.
            online_moire_margin (float): Margin ratio for center placement.
            **kwargs: Ignored arguments (allows passing full argparse dict).
        """
        self.size = size
        self.freq_range = np.arange(online_moire_freq_min, online_moire_freq_max + 1, dtype=np.int32)
        self.centers_min = online_moire_centers_min
        self.centers_max = online_moire_centers_max
        self.margin = size * online_moire_margin

        # --- Optimization: Precompute Grid ---
        # Computing meshgrid for every image is expensive. Do it once here.
        y = np.arange(size, dtype=np.float32)
        x = np.arange(size, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x, y, indexing='xy')
        self.scale = (2.0 * math.pi) / float(size)
        self.last_info = None

    def generate(self, return_info: bool = False):
        """
        Generate a random Moire pattern image using pre-computed grid.
        """
        # 1. Randomize parameters
        centers = np.random.randint(self.centers_min, self.centers_max + 1)
        freqs = np.random.choice(self.freq_range, size=centers, replace=False)
        
        z = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 2. Accumulate sine waves
        for f in freqs:
            cx = np.random.uniform(self.margin, self.size - self.margin)
            cy = np.random.uniform(self.margin, self.size - self.margin)
            
            # Distance calculation (bottleneck, but fast enough with numpy on CPU)
            dx = self.xx - cx
            dy = self.yy - cy
            r = np.sqrt(dx*dx + dy*dy)
            
            z += np.sin(self.scale * f * r)

        # 3. Normalize and convert to Image
        z /= float(centers)
        zmin, zmax = z.min(), z.max()
        
        if zmax > zmin + 1e-12:
            zn = (z - zmin) / (zmax - zmin)
            img_arr = (zn * 255.0 + 0.5).astype(np.uint8)
        else:
            img_arr = np.zeros_like(z, dtype=np.uint8)

        info = {
            "centers": int(centers),
            "freqs": [int(f) for f in np.asarray(freqs).tolist()],
        }
        self.last_info = info

        # Return as RGB for PixMix compatibility
        image = Image.fromarray(img_arr, mode='L').convert('RGB')
        if return_info:
            return image, info
        return image

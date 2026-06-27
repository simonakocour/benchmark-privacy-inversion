from itertools import combinations
import numpy as np
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity

def _rgb2gray(img: np.ndarray) -> np.ndarray:
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

class _VariabilityAccumulator:

    def __init__(self, device):
        self.device = device
        self.lpips_fn = LPIPS().to(device)
        self.lpips_fn.eval()

        self.pairwise_records = []
        self._image_count = 0

    def update(self, images: list):
        """
        images : list of (H, W, 3) float32 numpy arrays in [0, 1]
        """
        arrays = np.stack(images)                         # (B, H, W, 3)
        gray   = _rgb2gray(arrays)                        # (B, H, W) float [0,1]

        pairs = list(combinations(range(len(images)), 2))

        ssims = [
            structural_similarity(gray[i], gray[j], data_range=1.0)
            for i, j in pairs
        ]

        lp_vals = []
        for i, j in pairs:
            t1 = torch.from_numpy(gray[i]).unsqueeze(0).unsqueeze(0).float()
            t2 = torch.from_numpy(gray[j]).unsqueeze(0).unsqueeze(0).float()
            t1 = (t1.repeat(1, 3, 1, 1) * 2 - 1).to(self.device)
            t2 = (t2.repeat(1, 3, 1, 1) * 2 - 1).to(self.device)
            with torch.no_grad():
                lp_vals.append(self.lpips_fn(t1, t2).item())

        for k, (i, j) in enumerate(pairs):
            self.pairwise_records.append({
                "image_idx": self._image_count,
                "sample_i": i, "sample_j": j,
                "ssim": ssims[k], "lpips": lp_vals[k],
            })
        self._image_count += 1

    def summarize(self):
        all_ssims = [r["ssim"]  for r in self.pairwise_records]
        all_lpips = [r["lpips"] for r in self.pairwise_records]
        agg = {
            "mean_pairwise_ssim":  np.mean(all_ssims),
            "mean_pairwise_lpips": np.mean(all_lpips),
        }
        return agg, self.pairwise_records

def compute_variability_metrics(variability_out: dict, device: str = "cpu"):
    acc = _VariabilityAccumulator(torch.device(device))
    for images in variability_out.values():
        acc.update(images)
    return acc.summarize()

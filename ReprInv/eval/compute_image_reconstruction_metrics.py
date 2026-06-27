import numpy as np
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity
from skimage.transform import resize
from torchmetrics.image.fid import FrechetInceptionDistance


def _gray_lpips_pair(pred_np: np.ndarray, gt_np: np.ndarray, device):
    """Return (pred, gt) LPIPS tensors [1,3,H,W] in [-1, 1] (grayscale replicated)."""
    pred_rgb = torch.from_numpy(pred_np).permute(2, 0, 1).to(device)           # [3,H,W] float [0,1]
    gt_rgb = torch.from_numpy(gt_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)

    if pred_rgb.shape[-2:] != gt_rgb.shape[-2:]:
        gt_rgb = torch.nn.functional.interpolate(
            gt_rgb[None], size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False
        )[0]

    w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=pred_rgb.dtype)[:, None, None]
    pred_gray = (pred_rgb * w).sum(0, keepdim=True)
    gt_gray   = (gt_rgb   * w).sum(0, keepdim=True)

    return (pred_gray.repeat(3, 1, 1)[None] * 2 - 1,
            gt_gray.repeat(3, 1, 1)[None]  * 2 - 1)


class _ReconstructionAccumulator:

    def __init__(self, device):
        self.device = device
        self.lpips_fn = LPIPS().to(device)
        self.lpips_fn.eval()
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        self.ssims, self.lpips_vals = [], []

    def update(self, pred_np: np.ndarray, gt_np: np.ndarray):
        """
        pred_np : (H, W, 3) float32 in [0, 1]
        gt_np   : (H, W, 3) uint8
        """
        gt_f = resize(gt_np, pred_np.shape[:2], anti_aliasing=True)  # float [0,1]

        self.ssims.append(structural_similarity(gt_f, pred_np, channel_axis=2, data_range=1.0))

        pl, gl = _gray_lpips_pair(pred_np, gt_np, self.device)
        with torch.no_grad():
            self.lpips_vals.append(self.lpips_fn(pl, gl).item())

        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float().clamp(0, 1)
        gt_t   = torch.from_numpy(gt_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if pred_t.shape[-2:] != gt_t.shape[-2:]:
            gt_t = torch.nn.functional.interpolate(
                gt_t, size=pred_t.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0, 1)
        self.fid.update(gt_t,   real=True)
        self.fid.update(pred_t, real=False)

    def summarize(self):
        n = len(self.ssims)
        agg = {
            "SSIM":       np.mean(self.ssims),
            "LPIPS":      np.mean(self.lpips_vals),
            "FID":        self.fid.compute().item(),
            "num_images": n,
        }
        per_image = [
            {"image_idx": i, "SSIM": self.ssims[i], "LPIPS": self.lpips_vals[i]}
            for i in range(n)
        ]
        return agg, per_image


def compute_image_reconstruction_metrics(recons_ims: list, gt_ims: list, device: str = "cpu"):
    acc = _ReconstructionAccumulator(torch.device(device))
    for pred_np, gt_np in zip(recons_ims, gt_ims):
        acc.update(pred_np, gt_np)
    return acc.summarize()

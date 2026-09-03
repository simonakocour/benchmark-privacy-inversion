import cv2
from pathlib import Path
from torchvision.transforms import Compose

from .transforms import Resize, NormalizeImage, PrepareForNet
from ..segloc.models_hierar import DPTSegmentationModel

DPT_BACKBONE = "vitb_rn50_384"  # dpt_hybrid
ADE20K_NUM_CLASSES = 150


def build_dpt_transform(net_w: int = 512, net_h: int = 512) -> Compose:
    return Compose([
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])


def load_dpt_model(weights_path: Path, device: str):
    net = DPTSegmentationModel(
        ADE20K_NUM_CLASSES,
        path=str(weights_path),
        backbone=DPT_BACKBONE,
    )
    net.eval()
    return net.to(device)

import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from submodules.segloc.loader import load_segloc_model, ADE_COLOR_PALETTE
from ..utils.data_util import calculate_centroids_from_mask

_HF_REPO = "mpietran/privacy_inversion"
_SEGLOC_HF_FILENAME = "submodules/segloc/segloc_weights.pth"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _collect_image_paths(source: Union[str, Path, list]) -> list:
    if isinstance(source, list):
        return [Path(p) for p in source]
    source = Path(source)
    if source.is_file():
        return [source]
    paths = []
    for ext in _IMG_EXTS:
        paths.extend(source.rglob(f"*{ext}"))
        paths.extend(source.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))

class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self,
                 resolution=512,
                 downsample_factor=8,
                 device="cuda"):

        self.resolution = resolution
        self.device = device
        self.downsample_factor = downsample_factor

        self.image_transforms = T.Compose([
            T.Resize((self.resolution, self.resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        self.image_transforms_mask = T.Compose([
            T.Resize((self.resolution // self.downsample_factor,
                      self.resolution // self.downsample_factor),
                     interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def tensor_centroids_from_mask(self, segmentation_map, max_centroids=8192):
        centroids = calculate_centroids_from_mask(segmentation_map)
        shape = segmentation_map.size
        centroids = centroids / np.array([shape[0], shape[1]])
        n_centroids = centroids.shape[0]
        if n_centroids > max_centroids:
            centroids = np.random.permutation(centroids)[:max_centroids]
        return torch.from_numpy(centroids)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class SegLocImageDataset(BaseSegmentationDataset):
    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        n_clusters: int = 100,
        resolution: int = 512,
        downsample_factor: int = 8,
        device: str = "cuda",
    ):
        super().__init__(resolution=resolution, downsample_factor=downsample_factor, device=device)
        self.condition = condition
        self.image_paths = _collect_image_paths(source)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_SEGLOC_HF_FILENAME, token=token)
        self._model = load_segloc_model(Path(weights_path), n_clusters, device)
        self._infer_transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
        ])

    def _extract_segloc(self, image: Image.Image) -> Image.Image:
        img_t = self._infer_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sem_maps = self._model(img_t)[0]
        sem_map = sem_maps[-1].cpu().numpy().squeeze()
        labels = np.argmax(sem_map, axis=0)
        rgb = ADE_COLOR_PALETTE[labels].astype(np.uint8)
        return Image.fromarray(rgb)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        image_segmentation = self._extract_segloc(image)
        condition = self.image_transforms_mask(image_segmentation)
        positions = self.tensor_centroids_from_mask(image_segmentation).to(dtype=torch.float32)

        return {
            "image_path": str(image_path),
            "input_image": np.array(image),
            "condition": condition,
            "positions": positions,
        }

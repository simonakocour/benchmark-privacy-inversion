import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from submodules.segloc.loader import load_segloc_model, ADE_COLOR_PALETTE
from submodules.dpt.loader import load_dpt_model, build_dpt_transform
from ..utils.data_util import calculate_centroids_from_mask
from .image_datasets_sparse_descriptors import _collect_image_paths

_HF_REPO = "mpietran/privacy_inversion"
_SEGLOC_HF_FILENAME = "submodules/segloc/segloc_weights.pth"
_DPT_HF_FILENAME = "submodules/dpt/dpt_hybrid-ade20k-53898607.pt"
_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
_SAM2_MASK_GENERATOR_KWARGS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.5,
    "stability_score_thresh": 0.5,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 2,
}

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
            "segmentation_map": np.array(image_segmentation),
        }


class DptImageDataset(BaseSegmentationDataset):
    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        weights_path: Optional[Union[str, Path]] = None,
        resolution: int = 512,
        downsample_factor: int = 8,
        device: str = "cuda",
    ):
        super().__init__(resolution=resolution, downsample_factor=downsample_factor, device=device)
        self.condition = condition
        self.image_paths = _collect_image_paths(source)
        if weights_path is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_DPT_HF_FILENAME, token=token)
        self._model = load_dpt_model(Path(weights_path), device)
        self._infer_transform = build_dpt_transform(resolution, resolution)

    def _extract_dpt(self, image: Image.Image) -> Image.Image:
        img = np.array(image).astype(np.float32) / 255.0
        img_input = self._infer_transform({"image": img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            logits = self._model(sample)[1]
            labels = (torch.argmax(logits, dim=1) + 1).squeeze().cpu().numpy()
        rgb = ADE_COLOR_PALETTE[labels].astype(np.uint8)
        return Image.fromarray(rgb)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        image_segmentation = self._extract_dpt(image)
        condition = self.image_transforms_mask(image_segmentation)
        positions = self.tensor_centroids_from_mask(image_segmentation).to(dtype=torch.float32)

        return {
            "image_path": str(image_path),
            "input_image": np.array(image),
            "condition": condition,
            "positions": positions,
            "segmentation_map": np.array(image_segmentation),
        }


class SamImageDataset(BaseSegmentationDataset):
    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        resolution: int = 512,
        downsample_factor: int = 8,
        device: str = "cuda",
        model_id: str = _SAM2_MODEL_ID,
        checkpoint: Optional[Union[str, Path]] = None,
        model_cfg: Optional[str] = None,
        mask_generator_kwargs: Optional[dict] = None,
    ):
        super().__init__(resolution=resolution, downsample_factor=downsample_factor, device=device)
        self.condition = condition
        self.image_paths = _collect_image_paths(source)

        try:
            from sam2.build_sam import build_sam2, build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "SamImageDataset requires SAM 2: pip install "
                "'git+https://github.com/facebookresearch/sam2.git'"
            ) from exc

        if checkpoint is not None:
            if model_cfg is None:
                raise ValueError("model_cfg is required when checkpoint is given.")
            sam2 = build_sam2(model_cfg, str(checkpoint), device=device, apply_postprocessing=False)
        else:
            sam2 = build_sam2_hf(model_id, device=device, apply_postprocessing=False)

        kwargs = dict(_SAM2_MASK_GENERATOR_KWARGS)
        if mask_generator_kwargs:
            kwargs.update(mask_generator_kwargs)
        self._mask_generator = SAM2AutomaticMaskGenerator(sam2, **kwargs)

    def _extract_sam(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        if img.shape[:2] != (self.resolution, self.resolution):
            img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        masks = self._mask_generator.generate(img)

        rgb = np.ones((self.resolution, self.resolution, 3), dtype=np.float32)
        for ann in sorted(masks, key=lambda x: x["area"], reverse=True):
            rgb[ann["segmentation"]] = np.random.random(3)
        return Image.fromarray((rgb * 255).astype(np.uint8))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        image_segmentation = self._extract_sam(image)
        condition = self.image_transforms_mask(image_segmentation)
        positions = self.tensor_centroids_from_mask(image_segmentation).to(dtype=torch.float32)

        return {
            "image_path": str(image_path),
            "input_image": np.array(image),
            "condition": condition,
            "positions": positions,
            "segmentation_map": np.array(image_segmentation),
        }

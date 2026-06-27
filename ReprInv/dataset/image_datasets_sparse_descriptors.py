from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from submodules.superpoint.superpoint_fs import SuperPoint

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DESCRIPTOR_DIMS = {"SIFT": 128, "SUPERPOINT": 256, "XFEAT": 64}


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


class BaseSparseDescriptorDataset(Dataset, ABC):
    def __init__(self,
                 resolution=512,
                 downsample_factor=8,
                 noise_std_dev=0.0,
                 max_keypoints_extraction=8192,
                 num_keypoints_training_mid=1024):

        self.resolution = resolution
        self.downsample_factor = downsample_factor
        self.noise_std_dev = noise_std_dev
        self.max_keypoints_extraction = max_keypoints_extraction
        self.num_keypoints_training_mid = num_keypoints_training_mid

        self.xfeat = None
        self.superpoint = None

        self.image_transforms = T.Compose([
            T.Resize((self.resolution, self.resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        self.image_transforms_superpoint = T.Compose([
            T.Resize((self.resolution, self.resolution)),
            T.ToTensor(),
            T.Grayscale(num_output_channels=1),
        ])

        self.extractor = {
            'SIFT': self.extract_SIFT,
            'SUPERPOINT': self.extract_SUPERPOINT,
            'XFEAT': self.extract_XFEAT,
        }

    _extractors = {
        "SIFT": "extract_SIFT",
        "SUPERPOINT": "extract_SUPERPOINT",
        "XFEAT": "extract_XFEAT",
    }

    def _extract(self, image: Image.Image):
        return getattr(self, self._extractors[self.condition])(image)

    def _init_xfeat(self):
        if self.xfeat is None:
            self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat',
                                        pretrained=True, top_k=self.max_keypoints_extraction)
            self.xfeat.to("cpu")
            self.xfeat.dev = "cpu"
            self.xfeat.net = self.xfeat.net.to("cpu")
            self.xfeat.eval()

    def _init_superpoint(self):
        if self.superpoint is None:
            self.superpoint = SuperPoint()
            self.superpoint.to("cpu")
            self.superpoint.eval()

    @torch.no_grad()
    def extract_XFEAT(self, input_image):
        self._init_xfeat()
        input_image_torch = self.image_transforms(input_image)[None].cpu()
        out = self.xfeat.detectAndCompute(input_image_torch, top_k=self.max_keypoints_extraction)[0]
        return out['keypoints'].cpu().numpy(), out['descriptors'].cpu().numpy()

    @torch.no_grad()
    def extract_SIFT(self, input_image):
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints_extraction, contrastThreshold=0.04, edgeThreshold=20)
        input_image_cv = np.array(input_image.convert("L"))
        if input_image_cv.shape[0] != self.resolution or input_image_cv.shape[1] != self.resolution:
            input_image_cv = cv2.resize(input_image_cv, (self.resolution, self.resolution),
                                        interpolation=cv2.INTER_LINEAR)
        if input_image_cv.dtype != np.uint8:
            input_image_cv = (input_image_cv * 255).astype(np.uint8)

        keypoints_cv = sift.detect(input_image_cv, None)
        for kp in keypoints_cv:
            kp.angle = 0
        keypoints_cv, descriptors = sift.compute(input_image_cv, keypoints_cv)
        if descriptors is None:
            return np.zeros((self.max_keypoints_extraction, 2)), np.zeros((self.max_keypoints_extraction, 128))
        eps = 1e-7
        descriptors = descriptors / (np.sum(descriptors, axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        keypoints = np.array([kp.pt for kp in keypoints_cv], dtype=np.float32)
        return keypoints, descriptors

    @torch.no_grad()
    def extract_SUPERPOINT(self, input_image):
        self._init_superpoint()
        input_image_torch = self.image_transforms_superpoint(input_image)[None]
        out = self.superpoint({"image": input_image_torch})
        return out['keypoints'][0].numpy(), out['descriptors'][0].numpy().T

    @torch.no_grad()
    def create_conditioning_tensor(self, keypoints, descriptors, mode='direct',
                                   training=False, aggregate=False):
        h = w = self.resolution
        eps = 1e-7

        if training:
            random_factor = np.random.rand()
            num_keypoints = self.num_keypoints_training_mid // 2
            num_keypoints = int(num_keypoints * (1 + random_factor))
            idxs = np.random.permutation(len(keypoints))[:num_keypoints]
            keypoints = keypoints[idxs]
            descriptors = descriptors[idxs]
            
        if isinstance(descriptors, torch.Tensor):
            descriptors = descriptors.cpu().numpy()

        descriptors = (descriptors - descriptors.min(axis=1, keepdims=True)) / (
            descriptors.max(axis=1, keepdims=True) - descriptors.min(axis=1, keepdims=True) + eps)
        descriptors = descriptors * 2 - 1

        if self.noise_std_dev > 0:
            noise = np.random.normal(0, self.noise_std_dev, keypoints.shape)
            keypoints = keypoints + noise
            keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)

        if mode == 'sum_pool':
            dim = descriptors.shape[-1]
            conditioning_tensor = torch.zeros((dim, h, w))
            for kp, desc in zip(keypoints, descriptors):
                x, y = int(kp[0]), int(kp[1])
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                conditioning_tensor[:, y, x] = torch.tensor(desc, dtype=torch.float32).clone()
            conditioning_tensor = torch.nn.functional.avg_pool2d(
                conditioning_tensor,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            ) * (self.downsample_factor ** 2)
        elif mode == 'direct':
            dim = descriptors.shape[-1]
            lh, lw = h // self.downsample_factor, w // self.downsample_factor
            conditioning_tensor = torch.zeros((dim, lh, lw))
            if aggregate:
                count_t = torch.zeros((lh, lw))
                for kp, desc in zip(keypoints, descriptors):
                    x, y = int(kp[0] // self.downsample_factor), int(kp[1] // self.downsample_factor)
                    if x < 0 or y < 0 or x >= lw or y >= lh:
                        continue
                    conditioning_tensor[:, y, x] += torch.tensor(desc, dtype=torch.float32)
                    count_t[y, x] += 1.0
                mask = count_t > 0
                conditioning_tensor[:, mask] /= count_t[mask]
            else:
                for kp, desc in zip(keypoints, descriptors):
                    x, y = int(kp[0] // self.downsample_factor), int(kp[1] // self.downsample_factor)
                    if x < 0 or y < 0 or x >= lw or y >= lh:
                        continue
                    conditioning_tensor[:, y, x] = torch.tensor(desc, dtype=torch.float32).clone()

        if len(keypoints) > self.max_keypoints_extraction:
            keypoints = np.random.permutation(keypoints)[:self.max_keypoints_extraction]

        keypoints = torch.from_numpy(keypoints[:, :2].copy()).to(conditioning_tensor.dtype).clone()
        keypoints /= torch.tensor([h, w], dtype=conditioning_tensor.dtype)
        
        return conditioning_tensor, keypoints

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class SparseDescriptorImageDataset(BaseSparseDescriptorDataset):
    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        resolution: int = 512,
        downsample_factor: int = 8,
        aggregate: bool = False,
    ):
        if condition not in DESCRIPTOR_DIMS:
            raise ValueError(f"Unknown condition '{condition}'. Choose from {list(DESCRIPTOR_DIMS)}.")
        super().__init__(resolution=resolution,downsample_factor=downsample_factor)
        self.condition = condition
        self.aggregate = aggregate
        self.image_paths = _collect_image_paths(source)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        keypoints, descriptors = self._extract(image)
        condition, positions = self.create_conditioning_tensor(
            keypoints, descriptors, mode='direct', aggregate=self.aggregate
        )

        return {
            "image_path": str(path),
            "input_image": np.array(image),
            "condition": condition,     # (D, L, L)
            "positions": positions,     # (N, 2) in [0, 1]
        }

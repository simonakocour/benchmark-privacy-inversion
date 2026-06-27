import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from ReprInv.utils.data_util import calculate_centroids_from_mask  # used in prepare_segmentation
from inference import load_model

SEGMENTATION_CONDITIONS = ["segloc", "sam", "gsff", "adek", "ppnesf"]
DESCRIPTOR_CONDITIONS = ["SIFT", "SUPERPOINT", "XFEAT"]
PRIVACY_CONDITIONS = ["AASE_SIFT", "AASE_SUPERPOINT"]
NINJADESC_CONDITIONS = ["SIFT_ninjadesc"]
AUGMENTED_CONDITIONS = ["SIFT_permutation", "SIFT_random_lines", "SUPERPOINT_permutation", "SUPERPOINT_random_lines"]
DESCRIPTOR_DIMS = {
    "SIFT": 128, "SUPERPOINT": 256, "XFEAT": 64,
    "AASE_SIFT": 128, "AASE_SUPERPOINT": 256,
    "SIFT_ninjadesc": 128,
    "SIFT_permutation": 128, "SIFT_random_lines": 128,
    "SUPERPOINT_permutation": 256, "SUPERPOINT_random_lines": 256,
}

_CONFIG_MAP = {
    **{c: "configs/models/segmentation_diffusion.yaml" for c in SEGMENTATION_CONDITIONS},
    **{c: "configs/models/descriptor_diffusion.yaml" for c in DESCRIPTOR_CONDITIONS},
    **{c: "configs/models/descriptor_diffusion.yaml" for c in AUGMENTED_CONDITIONS},
    **{c: "configs/models/sparse_descriptors_obfuscation_diffusion.yaml" for c in PRIVACY_CONDITIONS},
    **{c: "configs/models/ninjadesc_diffusion.yaml" for c in NINJADESC_CONDITIONS},
}

def prepare_segmentation(
    mask_path: Path,
    resolution: int = 512,
    downsample_factor: int = 8,
    max_centroids:int = 8192,
) -> tuple:
    
    latent = resolution // downsample_factor
    mask = Image.open(mask_path).convert("RGB")

    transform = T.Compose([
        T.Resize((latent, latent), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    condition = transform(mask).unsqueeze(0)  # (1, 3, L, L)

    w, h = mask.size  
    centroids = calculate_centroids_from_mask(mask)  # (K, 2) pixel (x, y)

    if len(centroids) > max_centroids:
            centroids = np.random.permutation(centroids)[:max_centroids]

    centroids = centroids.astype(np.float32)
    centroids[centroids[:, 0] != -1, 0] /= w
    centroids[centroids[:, 0] != -1, 1] /= h

    positions = torch.from_numpy(centroids).unsqueeze(0)  # (1, N_centroids, 2)
    return condition, positions


def prepare_descriptors(
    npz_path: Path,
    condition: str,
    resolution: int = 512,
    downsample_factor: int = 8,
    aggregate: bool = False,
    max_keypoints: int = 8192
) -> tuple:
    
    data = np.load(npz_path)
    keypoints = data["keypoints"].astype(np.float32)      # (N, 2)  pixel (x, y)
    descriptors = data["descriptors"].astype(np.float32)  # (N, D)

    dim = DESCRIPTOR_DIMS[condition]
    if descriptors.shape[1] != dim:
        raise ValueError(
            f"Descriptor dimension mismatch for {condition}: "
            f"expected {dim}, got {descriptors.shape[1]}"
        )

    eps = 1e-7
    d_min = descriptors.min(axis=1, keepdims=True)
    d_max = descriptors.max(axis=1, keepdims=True)
    descriptors = (descriptors - d_min) / (d_max - d_min + eps) * 2.0 - 1.0

    latent_resolution = resolution // downsample_factor 
    cond_t = torch.zeros((dim, latent_resolution, latent_resolution), dtype=torch.float32)
    if aggregate:
        count_t = torch.zeros((latent_resolution, latent_resolution), dtype=torch.float32)
        for (x, y), desc in zip(keypoints, descriptors):
            xi = int(x / downsample_factor)
            yi = int(y / downsample_factor)
            if 0 <= xi < latent_resolution and 0 <= yi < latent_resolution:
                cond_t[:, yi, xi] += torch.from_numpy(desc)
                count_t[yi, xi] += 1.0
        mask = count_t > 0
        cond_t[:, mask] /= count_t[mask]
    else:
        for (x, y), desc in zip(keypoints, descriptors):
            xi = int(x / downsample_factor)
            yi = int(y / downsample_factor)
            if 0 <= xi < latent_resolution and 0 <= yi < latent_resolution:
                cond_t[:, yi, xi] = torch.from_numpy(desc)
    condition_tensor = cond_t.unsqueeze(0)  # (1, D, L, L)

    if len(keypoints) > max_keypoints:
        keypoints = np.random.permutation(keypoints)[:max_keypoints]

    kpts = keypoints[:, :2].copy()
    kpts = kpts / (resolution, resolution)  
    kpts_t = torch.from_numpy(kpts)
    kpts_t = kpts_t.unsqueeze(0)  # (1, num_keypoints, 2)

    return condition_tensor, kpts_t

def run_inversion(pipe, condition, conditioning_tensor, positions, device, **pipe_kwargs):
    conditioning_tensor = conditioning_tensor.to(device)
    positions = positions.to(device)

    pipe.set_condition(condition)
    pipe.eval()

    with torch.no_grad():
        image = pipe(conditioning_tensor, positions, **pipe_kwargs)

    return image

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().squeeze(0).permute(1, 2, 0).float().numpy()
    return Image.fromarray((arr * 255).round().clip(0, 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(
        description="Invert a privacy representation back to an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--condition", required=True, choices=list(_CONFIG_MAP))
    parser.add_argument("--input", required=True, help="Input file: PNG/JPG segmentation mask, or .npz descriptor file")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for diffusion")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--schedule_type", choices=["default", "spatial", "adaptive"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    inversion_model_config = _CONFIG_MAP[args.condition]

    with open(inversion_model_config) as f:
        config = yaml.safe_load(f)
    if args.schedule_type:
        config["schedule_type"] = args.schedule_type
    resolution = config.get("resolution", 512)
    downsample_factor = config.get("downsample_factor", 8)

    if args.condition in SEGMENTATION_CONDITIONS:
        print(f"Segmentation input: {input_path}")
        conditioning_tensor, positions = prepare_segmentation(input_path, resolution, downsample_factor)
    else:
        print(f"Descriptor input: {input_path}")
        conditioning_tensor, positions = prepare_descriptors(input_path, args.condition, resolution, downsample_factor)

    print(f"Loading model [{config.get('name')}]  condition={args.condition}")
    model = load_model(config, args.condition, args.device)

    pipe_kwargs = dict(
        num_inference_steps=args.num_inference_steps or config.get("num_inference_steps", 25),
        guidance_scale=args.guidance_scale or config.get("guidance_scale", 1.5),
        seed=args.seed,
        dtype=torch.float32,
    )

    image = run_inversion(model, args.condition, conditioning_tensor, positions, args.device, **pipe_kwargs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

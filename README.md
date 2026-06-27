# Vulnerability of Privacy-Preserving Visual Localization against Diffusion-based Attacks

## Environment setup

Requires Python 3.10 and CUDA 12.1.

```bash
conda create -n reprInv python=3.10
conda activate reprInv

# PyTorch (adjust cuda version if needed)
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
pip install torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

## Models

Four inversion models are available, each associated with a different type of input representation:

| Model | Config | Conditions |
|---|---|---|
| **Sparse descriptor inversion** | `descriptor_diffusion.yaml` | `SIFT`, `SUPERPOINT`, `XFEAT`, `SIFT_permutation`, `SIFT_random_lines`, `SUPERPOINT_permutation`, `SUPERPOINT_random_lines` |
| **AASE descriptor inversion** | `sparse_descriptors_obfuscation_diffusion.yaml` | `AASE_SIFT`, `AASE_SUPERPOINT` |
| **NinjaDesc inversion** | `ninjadesc_diffusion.yaml` | `SIFT_ninjadesc` |
| **Segmentation inversion** | `segmentation_diffusion.yaml` | `segloc`, `sam`, `gsff`, `adek`, `ppnesf` |

## Inference

### inversion.py — from image-based conditioning representations

Takes a pre-extracted image-based conditioning representation file and inverts it back to an image. The diffusion inversion model config is selected automatically based on the condition type.

**Segmentation map** (PNG/JPG image mask):

```bash
python inversion.py \
    --condition condition \
    --input /path/to/mask.png \
    --output reconstructed.png
```

Supported `--condition` values: `segloc`, `sam`, `gsff`, `adek`, `ppnesf`.

**Sparse descriptors** (`.npz` file):
The `.npz` file must contain:

| Key | Shape | Description |
|---|---|---|
| `keypoints` | `(N, 2)` float32 | Pixel `(x, y)` coordinates |
| `descriptors` | `(N, D)` float32 | Feature vectors |

```bash
python inversion.py \
    --condition condition \
    --input /path/to/features.npz \
    --output reconstructed.png
```

Supported `--condition` values: `SIFT`, `SUPERPOINT`, `XFEAT`, `SIFT_permutation`, `SIFT_random_lines`, `SUPERPOINT_random_lines`, `SUPERPOINT_permutation`, `AASE_SIFT`, `AASE_SUPERPOINT`, `SIFT_ninjadesc`

### inversion_from_images.py — from images

Takes a folder of images, extracts image-based conditioning representations, runs the inversion model, and computes privacy metrics with regard to ground truth input images.

```bash
python inversion_from_images.py \
    --condition condition \
    --input /path/to/images/ \
    --output results/
```

Supported `--condition` values: `SIFT`, `SUPERPOINT`, `XFEAT`, `segloc`

Additional options:
| Flag | Default | Description |
|---|---|---|
| `--num_inference_steps` | from config | Number of diffusion steps |
| `--guidance_scale` | from config | Classifier-free guidance scale |
| `--n_variability_seeds` | 10 | Seeds per image for variability metrics |
| `--skip_vlm` | False | Skip VLM-based metrics |

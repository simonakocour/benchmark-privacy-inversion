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

Takes a pre-extracted image-based conditioning representation file and inverts it back to an image. The diffusion inversion model config is selected automatically based on the condition type. Inversion model expects representation extracted from 512x512 images.

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

Supported `--condition` values: `SIFT` (upright rootSIFT), `SUPERPOINT`, `XFEAT`, `SIFT_permutation` (upright rootSIFT, geometrically deobfuscated from permutation), `SIFT_random_lines` (upright rootSIFT, geometrically deobfuscated from RL), `SUPERPOINT_random_lines` (geometrically deobfuscated from RL), `SUPERPOINT_permutation` (geometrically deobfuscated from permutation), `AASE_SIFT`, `AASE_SUPERPOINT`, `SIFT_ninjadesc`

### inversion_from_images.py — from images

Takes a folder of images, extracts image-based conditioning representations, runs the inversion model, and computes privacy metrics with regard to ground truth input images.

```bash
python inversion_from_images.py \
    --condition condition \
    --input /path/to/images/ \
    --output results/
```

Supported `--condition` values, with the extractor each one runs:

| Condition | Extractor | Model config |
|---|---|---|
| `SIFT`, `SUPERPOINT`, `XFEAT` | upright rootSIFT / SuperPoint / XFeat | `descriptor_diffusion.yaml` |
| `SIFT_ninjadesc` | SIFT keypoints + NinjaDesc privacy head | `ninjadesc_diffusion.yaml` |
| `segloc` | SegLoc | `segmentation_diffusion.yaml` |
| `sam` | SAM 2 | `segmentation_diffusion.yaml` |
| `adek` | DPT hybrid ADE20K | `segmentation_diffusion.yaml` |


Extractor weights are fetched on first use and cached from the HF repo `mpietran/privacy_inversion` for SegLoc, SuperPoint and DPT (set `HF_TOKEN` if it is private), from `facebook/sam2.1-hiera-tiny` for SAM 2, and from the public NinjaDesc release into `~/.cache/ninjadesc`. Only `--condition sam` needs a dependency beyond `requirements.txt`:

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Additional options:
| Flag | Default | Description |
|---|---|---|
| `--num_inference_steps` | from config | Number of diffusion steps |
| `--guidance_scale` | from config | Classifier-free guidance scale |
| `--n_variability_seeds` | 10 | Seeds per image for variability metrics |
| `--skip_vlm` | False | Skip VLM-based metrics |

## Training

### train.py

Trains the inversion model from a folder of images on a set of conditions. The conditioning representations are
extracted on the fly with the same code the inference scripts use, so no pre-extraction step is needed. The LoRA tasks and their input dimensions are derived from `--conditions`. 
When loading a trained inversion model, the list and order of conditions must match the one provided in training.

```bash
# single condition
python train.py \
    --train_images /path/to/train/ \
    --val_images /path/to/val/ \
    --conditions SIFT \
    --output_dir runs/sift

# several conditions in one model
python train.py \
    --train_images /path/to/train/ \
    --val_images /path/to/val/ \
    --conditions segloc sam adek \
    --output_dir runs/segmentation
```

Any condition below can be combined; `--conditions` accepts one or more.

| Condition | conv_in channels |
|---|---|
| `SIFT`, `SIFT_ninjadesc` | 128 |
| `SUPERPOINT` | 256 |
| `XFEAT` | 64 |
| `segloc`, `sam`, `adek` | 3 |

Additional representations not present in the list of available conditions may be added by writing and integrating an extraction dataset (following the structure of existing dataset classes).
The defaults arguments in train.py mirror the training parameters the released checkpoints were trained with. Only the LoRA layers, the per-task `convs_in` and the sparse pixel graph encoder are trained; the rest of the UNet and the VAE stay frozen.

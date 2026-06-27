import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ReprInv.dataset import SparseDescriptorImageDataset, SegLocImageDataset
from ReprInv.eval import compute_vlm_metrics, compute_image_reconstruction_metrics, compute_variability_metrics

from inference import load_model

SEGMENTATION_CONDITIONS = ["segloc", "sam", "gsff", "adek", "ppnesf"]
DESCRIPTOR_CONDITIONS = ["SIFT", "SUPERPOINT", "XFEAT"]
AASE_CONDITIONS = ["AASE_SIFT", "AASE_SUPERPOINT"]
_AASE_BASE = {"AASE_SIFT": "SIFT", "AASE_SUPERPOINT": "SUPERPOINT"}
DESCRIPTOR_DIMS = {"SIFT": 128, "SUPERPOINT": 256, "XFEAT": 64}

def run_inversion(pipe, dataloader, condition, device, **pipe_kwargs):
    pipe.set_condition(condition)
    pipe.eval()

    recons_ims, gt_ims = [], []
    for batch in dataloader:
        conditioning_tensor = batch["condition"][0:1].to(device)
        gt_image = batch["input_image"][0:1].to(device)
        positions = batch["positions"][0:1].to(device)

        with torch.no_grad():
            image = pipe(conditioning_tensor, positions, **pipe_kwargs)

        recons_ims.append(tensor_to_np(image))
        gt_ims.append(gt_image[0].cpu().numpy())

    return {"recons": recons_ims, "gt": gt_ims}


def run_inversion_variability(pipe, dataloader, condition, device, seeds=None, n_seeds_per_image=10, n_images=10, **pipe_kwargs):
    pipe.set_condition(condition)
    pipe.eval()

    pipe_kwargs_no_seed = {k: v for k, v in pipe_kwargs.items() if k != "seed"}

    out = {}
    for image_idx, batch in enumerate(dataloader):
        conditioning_tensor = batch["condition"][0:1].to(device)
        positions = batch["positions"][0:1].to(device)

        image_seeds = seeds or [np.random.randint(0, 1_000_000) for _ in range(n_seeds_per_image)]
        recons_ims = [
            tensor_to_np(pipe(conditioning_tensor, positions, seed=s, **pipe_kwargs_no_seed))
            for s in image_seeds
        ]
        out[image_idx] = recons_ims

        if image_idx >= n_images - 1:
            break

    return out

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().squeeze(0).permute(1, 2, 0).float().numpy()
    return Image.fromarray((arr * 255).round().clip(0, 255).astype(np.uint8))

def tensor_to_np(image: torch.Tensor) -> np.array:
    return image.detach().cpu().squeeze(0).permute(1, 2, 0).float().numpy()
    

def main():
    parser = argparse.ArgumentParser(
        description="Invert a privacy representation back to an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--condition", required=True, choices=SEGMENTATION_CONDITIONS + DESCRIPTOR_CONDITIONS + AASE_CONDITIONS)
    parser.add_argument("--input", required=True, help="Input image or folder of images")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diffusion")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--aggregate", action="store_true", help="Aggregate descriptors per latent cell")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM metrics (Qwen2-VL-7B)")
    parser.add_argument("--n_variability_seeds", type=int, default=10, help="Seeds per image for variability")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.condition in SEGMENTATION_CONDITIONS:
        dataset = SegLocImageDataset(args.input,args.condition)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)
        inversion_model_config = "segmentation_diffusion.yaml"

    elif args.condition in DESCRIPTOR_CONDITIONS:
        dataset = SparseDescriptorImageDataset(args.input, args.condition, aggregate=args.aggregate)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)
        inversion_model_config = "descriptor_diffusion.yaml"

    elif args.condition in AASE_CONDITIONS:
        dataset = SparseDescriptorImageDataset(args.input, _AASE_BASE[args.condition], privacy_descriptors=True)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)
        inversion_model_config = "sparse_descriptors_obfuscation_diffusion.yaml"

    else:
        raise ValueError("input modality not handled")

    ## load inversion model
    config_path = Path("configs/models") / inversion_model_config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Loading model [{config.get('name')}]  condition={args.condition}")
    model = load_model(config, args.condition, args.device)

    pipe_kwargs = dict(
        num_inference_steps=args.num_inference_steps or config.get("num_inference_steps", 25),
        guidance_scale=args.guidance_scale or config.get("guidance_scale", 1.5),
        seed=args.seed,
        dtype=torch.float32,
    )

    ## run inversion
    out = run_inversion(model, dataloader, args.condition, args.device, **pipe_kwargs)
    out_variability = run_inversion_variability(
        model, dataloader, args.condition, args.device,
        n_seeds_per_image=args.n_variability_seeds, **pipe_kwargs,
    )

    ## compute metrics
    recon_agg, recon_per_image = compute_image_reconstruction_metrics(
        out["recons"], out["gt"], device=args.device
    )
    var_agg, var_records = compute_variability_metrics(out_variability, device=args.device)
    if not args.skip_vlm:
        vlm_agg, vlm_per_image = compute_vlm_metrics(
            out["recons"], out["gt"], device=args.device
        )
    else:
        vlm_agg, vlm_per_image = {}, []

    ## save metrics and reconstructions
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (pred_np, gt_np) in enumerate(zip(out["recons"], out["gt"])):
        pred_pil = Image.fromarray((pred_np * 255).round().clip(0, 255).astype("uint8"))
        gt_pil   = Image.fromarray(gt_np)
        w, h = pred_pil.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(gt_pil.resize((w, h)), (0, 0))
        combined.paste(pred_pil, (w, 0))
        combined.save(out_dir / f"reconstruction_{i:04d}.png")

    def _write_metrics(path, metrics: dict):
        with open(path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved: {path}")

    def _write_tsv(path, rows: list):
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(
                    f"{v:.6f}" if isinstance(v, float) else str(v)
                    for v in row.values()
                ) + "\n")
        print(f"Saved: {path}")

    _write_metrics(out_dir / "reconstruction_metrics.txt", recon_agg)
    _write_tsv(out_dir / "reconstruction_per_image.tsv",   recon_per_image)
    _write_metrics(out_dir / "variability_metrics.txt",    var_agg)
    _write_tsv(out_dir / "variability_per_pair.tsv",       var_records)
    if not args.skip_vlm:
        _write_metrics(out_dir / "vlm_metrics.txt",        vlm_agg)
        _write_tsv(out_dir / "vlm_per_image.tsv",          vlm_per_image)
    print("Done.")

if __name__ == "__main__":
    main()

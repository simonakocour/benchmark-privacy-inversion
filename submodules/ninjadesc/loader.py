import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.hub

_RELEASE_BASE = "https://github.com/facebookresearch/ninjadesc/releases/download/v1.0"
WEIGHT_FILES = {
    # Stage-1 NinjaNet utility-init torchscripts.
    "ninja_desc_sosnet_init": "ninja_desc_sos_init_torchscript.pt",
    "ninja_desc_sift_init": "ninja_desc_sift_init_torchscript.pt",
    "ninja_desc_hardnet_init": "ninja_desc_hardnet_init_torchscript.pt",
    # Trained NinjaNet checkpoints from joint adversarial training (paper Table 2/3, lambda=1).
    "ninja_desc_sosnet_joint": "ninja_desc_sosnet_joint.ckpt",
    "ninja_desc_sift_joint": "ninja_desc_sift_joint.ckpt",
    "ninja_desc_hardnet_joint": "ninja_desc_hardnet_joint.ckpt",
    # Base descriptor weights.
    "sosnet_scape_pipeline": "sosnet_scape.pt",
    "hardnet_lib": "hardnet_lib.pt",
}

BASE_DESCRIPTORS = frozenset({"sosnet", "sift", "hardnet"})


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "ninjadesc"


def download_weights(name: str) -> str:
    if name not in WEIGHT_FILES:
        raise KeyError(f"Unknown weight name {name!r}. Known: {sorted(WEIGHT_FILES)}")
    filename = WEIGHT_FILES[name]
    cache_dir = _cache_root()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename
    if not local_path.exists():
        url = f"{_RELEASE_BASE}/{filename}"
        torch.hub.download_url_to_file(url, str(local_path))
    return str(local_path)


def fix_state_dict_keys(net_key: str, state_dict: OrderedDict) -> OrderedDict:
    fixed_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(net_key):
            k_new = k[len(net_key) + 1 :]
            fixed_state_dict[k_new] = v

    return fixed_state_dict


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D descriptors of shape (N, D), got {tuple(x.shape)}")
    return x


def load_ninjadesc_models(base_desc: str):
    base_desc = str(base_desc).lower()
    if base_desc not in BASE_DESCRIPTORS:
        raise ValueError(f"base_desc must be one of {sorted(BASE_DESCRIPTORS)}, got {base_desc!r}.")

    # ---- Privacy head ------------------------------------------------------
    # The joint checkpoints are Lightning .ckpt files holding a privacy.* state_dict;
    # the architecture itself comes from the stage-1 init TorchScript.
    init_ts = torch.jit.load(download_weights(f"ninja_desc_{base_desc}_init"), map_location="cpu")
    privacy = getattr(init_ts, "privacy", None)
    if privacy is None:
        raise RuntimeError(f"Init torchscript for {base_desc!r} has no .privacy submodule")

    ckpt = torch.load(download_weights(f"ninja_desc_{base_desc}_joint"), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    privacy.load_state_dict(fix_state_dict_keys(net_key="privacy", state_dict=state_dict))
    privacy = privacy.eval()

    # ---- Base descriptor model ---------------------------------------------
    if base_desc == "sift":
        from kornia.feature import SIFTDescriptor

        base = SIFTDescriptor(32, 8, 4).eval()
    elif base_desc == "sosnet":
        base = torch.jit.load(download_weights("sosnet_scape_pipeline"), map_location="cpu").eval()
    else:  # hardnet
        base = torch.jit.load(download_weights("hardnet_lib"), map_location="cpu").eval()

    return privacy, base

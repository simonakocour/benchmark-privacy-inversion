import torch
import math
from typing import Union


def _compute_sinusoidal_encoding(coords_norm: torch.Tensor, d_model: int, device: torch.device) -> torch.Tensor:
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(1000.0) / d_model))
    pe = torch.zeros(coords_norm.shape[0], d_model, device=device, dtype=coords_norm.dtype)
    pe[:, 0::2] = torch.sin(coords_norm * div_term)
    pe[:, 1::2] = torch.cos(coords_norm * div_term)
    return pe


def get_2d_positional_encoding(
    x: Union[torch.Tensor, float], 
    y: Union[torch.Tensor, float], 
    d_model_total: int, 
    max_w: int, 
    max_h: int, 
    device: torch.device
) -> torch.Tensor:
    
    if d_model_total % 2 != 0:
        raise ValueError("d_model_total must be an even number")
    d_model_half = d_model_total // 2
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float, device=device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float, device=device)

    x_norm = (x.to(device=device, dtype=torch.float) / max_w).unsqueeze(1)
    y_norm = (y.to(device=device, dtype=torch.float) / max_h).unsqueeze(1)

    pe_x = _compute_sinusoidal_encoding(x_norm, d_model_half, device)
    pe_y = _compute_sinusoidal_encoding(y_norm, d_model_half, device)

    pe = torch.cat([pe_x, pe_y], dim=1)
    pe = (pe - pe.mean(dim=1, keepdim=True)) / (pe.std(dim=1, keepdim=True) + 1e-6)
    
    return pe


def get_2d_positional_encoding_normalized(coords: torch.Tensor, d_model_total: int) -> torch.Tensor:
    if d_model_total % 2 != 0:
        raise ValueError("d_model_total must be an even number")
    d_model_half = d_model_total // 2
    
    device = coords.device
    x_norm = coords[:, 0].unsqueeze(1)
    y_norm = coords[:, 1].unsqueeze(1)

    pe_x = _compute_sinusoidal_encoding(x_norm, d_model_half, device)
    pe_y = _compute_sinusoidal_encoding(y_norm, d_model_half, device)

    pe = torch.cat([pe_x, pe_y], dim=1)
    pe = (pe - pe.mean(dim=1, keepdim=True)) / (pe.std(dim=1, keepdim=True) + 1e-6)
    
    return pe

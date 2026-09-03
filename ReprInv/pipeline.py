import torch
from tqdm.auto import tqdm
import torch.nn as nn
from typing import Optional, Tuple, Dict

from ReprInv.utils.data_util import knn_edge_index
from ReprInv.model.cond_lora_unet import UNet2DConditionModelConditionalLora
from diffusers import SchedulerMixin, AutoencoderKL


class Representation_Inversion_Diffusion(object):

    def __init__(self,
                 unet: UNet2DConditionModelConditionalLora,
                 vae: AutoencoderKL,
                 noise_scheduler: SchedulerMixin,
                 cfg_dropout_p: float = 0.1,
                 token_feature_dim: int = 1024,
                 init_token_norm_scale: float = 0.1,
                 resolution: int = 512):
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.cfg_dropout_p = cfg_dropout_p
        self.resolution = resolution

        self.token_norm = nn.LayerNorm(token_feature_dim)
        with torch.no_grad():
            self.token_norm.weight.fill_(init_token_norm_scale)
            self.token_norm.bias.zero_()
        
    def to(self, device: torch.device, weight_dtype: torch.dtype) -> None:
        self.unet.to(device, dtype=torch.float32)
        self.vae.to(device, dtype=weight_dtype)
        self.token_norm.to(device=device, dtype=torch.float32)
        
        self.device = device
        self.weight_dtype = weight_dtype
        self.mixed_precision = weight_dtype != torch.float32
    
    def set_condition(self, condition: str) -> None:
        if hasattr(self.unet, 'module'):
            self.unet.module.switch_lora(condition)
        else:
            self.unet.switch_lora(condition)

    def train(self) -> None:
        self.unet.train()

    def eval(self) -> None:
        self.unet.eval()

    @torch.no_grad()
    def create_noisy_latent_from_image(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latents = self.vae.encode(input_image.to(self.device, dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        if hasattr(self.noise_scheduler, 'sample_train_timesteps') and hasattr(self, '_last_positions'):
            timesteps = self.noise_scheduler.sample_train_timesteps(bsz, self._last_positions, latents.device)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
        timesteps = timesteps.long()
        
        noisy_latents = self.noise_scheduler.add_noise(
            latents.float(), noise.float(), timesteps
        ).to(torch.float32)

        noise_info = {
            "latents": latents,
            "timesteps": timesteps,
            "noise": noise
        }
        return noisy_latents, noise_info
    
    def create_crossatn_conditioning(
        self, 
        positions: torch.Tensor, 
        n_bins: int = 8,
        strip_padding: bool = False,
    ) -> torch.Tensor:

        self._last_positions = positions        
        encoder_hidden_states = []
        for position in positions:
            if strip_padding:
                position = position[position[:, 0] != -1]

            encoder = (
                self.unet.module.sparse_pixel_graph_encoder 
                if hasattr(self.unet, 'module') 
                else self.unet.sparse_pixel_graph_encoder
            )

            if position.shape[0] == 0:        
                encoder_hidden_states.append(torch.zeros((n_bins * n_bins, encoder.latent_dim), device=self.device, dtype=torch.float32))
                continue

            edge_index = knn_edge_index(position).to(self.device)
            
            tokens = encoder(
                position.to(self.device, dtype=torch.float32), 
                edge_index, 
                n_bins=n_bins
            )
            encoder_hidden_states.append(tokens)
            
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
        return encoder_hidden_states

    def compute_loss(self, model_pred: torch.Tensor, noise_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise_info["noise"]
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(noise_info["latents"], noise_info["noise"], noise_info["timesteps"])
        else:
            raise ValueError(
                f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}. "
                f"Supported types: 'epsilon', 'v_prediction'"
            )
        
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def forward_train(
        self, 
        input_image: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        condition = condition.to(self.device, dtype=self.weight_dtype)
        encoder_hidden_states = self.create_crossatn_conditioning(positions, strip_padding=True)
        noisy_latents, noise_info = self.create_noisy_latent_from_image(input_image)
    
        if self.cfg_dropout_p is not None and self.cfg_dropout_p > 0.0:
            bsz = encoder_hidden_states.shape[0]
            drop_mask = (torch.rand(bsz, device=self.device) < self.cfg_dropout_p)
            if drop_mask.any():
                encoder_hidden_states = encoder_hidden_states.clone()
                encoder_hidden_states[drop_mask] = 0
                
                # Also zero out local conditioning to match inference CFG
                condition = condition.clone()
                condition[drop_mask] = 0

        noisy_latents = torch.cat([noisy_latents, condition], dim=1)
        
        if self.mixed_precision:
            with torch.cuda.amp.autocast(enabled=True):
                model_pred = self.unet(noisy_latents,noise_info["timesteps"],encoder_hidden_states=encoder_hidden_states)[0]
        else:
            model_pred = self.unet(noisy_latents,noise_info["timesteps"],encoder_hidden_states=encoder_hidden_states)[0]

        loss = self.compute_loss(model_pred, noise_info)
        return loss
    
    @torch.no_grad()
    def __call__(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        downsample_factor: int = 8,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        if hasattr(scheduler, 'roi_blend'):
            scheduler.set_timesteps(num_inference_steps, positions=positions)
        else:
            scheduler.set_timesteps(num_inference_steps)

        cross_attn_conditioning = self.create_crossatn_conditioning(positions[0:1])
        cross_attn_conditioning = cross_attn_conditioning.to(self.device, dtype=dtype)
        condition = condition[0:1].to(self.device, dtype=dtype)

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        in_channels = (self.unet.module.config.in_channels if hasattr(self.unet, 'module') else self.unet.config.in_channels)
        latents = torch.randn(
            (1, in_channels, self.resolution // downsample_factor, self.resolution // downsample_factor),
            generator=generator,
            device=self.device,
            dtype=dtype
        )
        latents = latents * scheduler.init_noise_sigma
        latents = torch.cat([latents, condition], dim=1)

        for t in tqdm(scheduler.timesteps, desc="Denoising"):
            latents_uncond = latents.clone()
            latents_uncond[:, 4:] = 0
            latents_cfg = torch.cat([latents_uncond, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latents_cfg, timestep=t)

            ehs_uncond = torch.zeros_like(cross_attn_conditioning)
            ehs_cfg = torch.cat([ehs_uncond, cross_attn_conditioning], dim=0)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=ehs_cfg).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents_up = scheduler.step(noise_pred, t, latents[:, :4]).prev_sample
            latents = torch.cat([latents_up, latents[:, 4:]], dim=1)
        
        image = self.vae.decode(latents[:, :4] / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image
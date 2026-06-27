
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from diffusers import DDPMScheduler
import numpy as np


class SpatialDDPMScheduler:    
    def __init__(
        self,
        base_scheduler: DDPMScheduler,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        # Spatial attention parameters
        sigma: float = 15.0,
        attention_mode: str = "frequency",  # "variance", "gradient", or "frequency"
        attention_strength: float = 0.1,  # Very subtle
        attention_warmup_steps: int = 200,  # Only apply after initial denoising
        attention_cooldown_steps: int = 100,  # Stop before final steps
        # Metric preservation
        local_stats_window: int = 32,  # Window size for local statistics
        global_stats_weight: float = 0.9,  # How much to preserve global stats
        device: str = "cuda",
        **kwargs
    ):
        
        self.base_scheduler = base_scheduler
        self.sigma = sigma
        self.attention_mode = attention_mode
        self.attention_strength = attention_strength
        self.attention_warmup_steps = attention_warmup_steps
        self.attention_cooldown_steps = attention_cooldown_steps
        self.local_stats_window = local_stats_window
        self.global_stats_weight = global_stats_weight
        self.device = device
        self._attention_cache = None
        self.base_scheduler.num_train_timesteps = num_train_timesteps

    def create_attention_map(
        self,
        keypoints: Union[torch.Tensor, List, np.ndarray],
        image_size: Tuple[int, int],
        batch_size: int = 1
    ) -> torch.Tensor:
        H, W = image_size
        
        if isinstance(keypoints, (list, np.ndarray)):
            keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints = keypoints.float().to(self.device)
        
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(0).repeat(batch_size, 1, 1)
        
        B, N, _ = keypoints.shape
        

        y = torch.arange(H, device=self.device).float().view(1, 1, H, 1)
        x = torch.arange(W, device=self.device).float().view(1, 1, 1, W)
        attention = torch.zeros((B, 1, H, W), device=self.device)        
        for i in range(N):
            kp_x = keypoints[:, i, 0:1].view(B, 1, 1, 1)
            kp_y = keypoints[:, i, 1:2].view(B, 1, 1, 1)
          
            if kp_x.max() <= 1.0 and kp_y.max() <= 1.0:
                kp_x = kp_x * W
                kp_y = kp_y * H
            dist_sq = ((x - kp_x) / W)**2 + ((y - kp_y) / H)**2
            gaussian = torch.exp(-dist_sq * (W + H) / (4 * self.sigma))
            
            attention = torch.max(attention, gaussian)
        attention = (attention - attention.mean()) / (attention.std() + 1e-8)
        attention = torch.sigmoid(attention)
        
        return attention
    
    def compute_local_statistics(
        self,
        tensor: torch.Tensor,
        window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = tensor.shape
        kernel = window_size
        stride = window_size // 2
        padding = kernel // 2
        local_mean = F.avg_pool2d(tensor, kernel, stride=1, padding=padding)
        squared = F.avg_pool2d(tensor ** 2, kernel, stride=1, padding=padding)
        local_var = squared - local_mean ** 2
        local_std = torch.sqrt(local_var + 1e-8)
        
        return local_mean, local_std
    
    def apply_variance_guidance(
        self,
        model_output: torch.Tensor,
        attention: torch.Tensor,
        strength: float
    ) -> torch.Tensor:

        B, C, H, W = model_output.shape
        

        local_mean, local_std = self.compute_local_statistics(
            model_output, self.local_stats_window
        )
        

        global_mean = model_output.mean(dim=[2, 3], keepdim=True)
        global_std = model_output.std(dim=[2, 3], keepdim=True)
        target_std = local_std * (1 + strength * (attention - 0.5))
        
        normalized = (model_output - local_mean) / (local_std + 1e-8)
        modulated = normalized * target_std + local_mean
        final_output = (self.global_stats_weight * model_output + 
                       (1 - self.global_stats_weight) * modulated)
        final_output = final_output - final_output.mean(dim=[2, 3], keepdim=True) + global_mean
        return final_output
    
    def apply_gradient_guidance(
        self,
        model_output: torch.Tensor,
        attention: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        B, C, H, W = model_output.shape
        
        grad_x = model_output[:, :, :, 1:] - model_output[:, :, :, :-1]
        grad_y = model_output[:, :, 1:, :] - model_output[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        gradient_weight = 1 + strength * (attention - 0.5)
        sharpened = model_output + gradient_weight * (grad_x + grad_y) * 0.1
       
        global_mean = model_output.mean(dim=[2, 3], keepdim=True)
        global_std = model_output.std(dim=[2, 3], keepdim=True) 
        sharpened_mean = sharpened.mean(dim=[2, 3], keepdim=True)
        sharpened_std = sharpened.std(dim=[2, 3], keepdim=True)
        
        final_output = (sharpened - sharpened_mean) / (sharpened_std + 1e-8)
        final_output = final_output * global_std + global_mean
        return final_output
    
    def apply_frequency_guidance(
        self,
        model_output: torch.Tensor,
        attention: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        B, C, H, W = model_output.shape
        
        low_freq = F.avg_pool2d(
            F.avg_pool2d(model_output, 3, stride=1, padding=1),
            3, stride=1, padding=1
        )
        high_freq = model_output - low_freq
        freq_weight = 1 + strength * (attention - 0.5)
        final_output = low_freq + freq_weight * high_freq
        
        global_mean = model_output.mean(dim=[2, 3], keepdim=True)
        global_std = model_output.std(dim=[2, 3], keepdim=True)
        
        final_mean = final_output.mean(dim=[2, 3], keepdim=True)
        final_std = final_output.std(dim=[2, 3], keepdim=True)
        
        final_output = (final_output - final_mean) / (final_std + 1e-8)
        final_output = final_output * global_std + global_mean
        
        return final_output
    
    def get_attention_weight(self, timestep: int, num_inference_steps: int) -> float:
        current_step = num_inference_steps - (timestep / self.base_scheduler.num_train_timesteps * num_inference_steps)
        if current_step < self.attention_warmup_steps:
            return 0.0
        
        if current_step > num_inference_steps - self.attention_cooldown_steps:
            return 0.0
        
        warmup_factor = min(1.0, (current_step - self.attention_warmup_steps) / 100)
        cooldown_factor = min(1.0, (num_inference_steps - self.attention_cooldown_steps - current_step) / 100)
        
        return self.attention_strength * warmup_factor * cooldown_factor
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        keypoints: Optional[Union[torch.Tensor, List]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        if keypoints is None:
            return self.base_scheduler.step(
                model_output, timestep, sample, 
                generator=generator, return_dict=return_dict, **kwargs
            )
        
        B, C, H, W = sample.shape
        
        num_inference_steps = len(self.timesteps) if hasattr(self, 'timesteps') else 50
        attention_weight = self.get_attention_weight(timestep, num_inference_steps)
        
        if attention_weight > 0:
            if self._attention_cache is None or self._attention_cache.shape[-2:] != (H, W):
                self._attention_cache = self.create_attention_map(keypoints, (H, W), B)
            attention = self._attention_cache
          
            if attention.shape[1] != C:
                attention = attention.expand(B, C, H, W)
            
            if self.attention_mode == "variance":
                guided_output = self.apply_variance_guidance(
                    model_output, attention, attention_weight
                )
            elif self.attention_mode == "gradient":
                guided_output = self.apply_gradient_guidance(
                    model_output, attention, attention_weight
                )
            elif self.attention_mode == "frequency":
                guided_output = self.apply_frequency_guidance(
                    model_output, attention, attention_weight
                )
            else:
                guided_output = model_output
            
            guided_output = torch.nan_to_num(guided_output, nan=0.0, posinf=1e10, neginf=-1e10)
            
        else:
            guided_output = model_output
        
        return self.base_scheduler.step(
            guided_output, timestep, sample,
            generator=generator, return_dict=return_dict, **kwargs
        )
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        keypoints: Optional[Union[torch.Tensor, List]] = None
    ) -> torch.Tensor:
        return self.base_scheduler.add_noise(original_samples, noise, timesteps)
    
    def set_timesteps(self, num_inference_steps: int, device=None):
        self.base_scheduler.set_timesteps(num_inference_steps, device)
        self.timesteps = self.base_scheduler.timesteps
        self._attention_cache = None

    @property
    def config(self):
        return self.base_scheduler.config
    
    @property
    def init_noise_sigma(self):
        return self.base_scheduler.init_noise_sigma
    
    @property
    def num_train_timesteps(self):
        return self.base_scheduler.num_train_timesteps
    
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return self.base_scheduler.scale_model_input(sample, timestep)
    
    def __getattr__(self, name):
        return getattr(self.base_scheduler, name)

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .lora_layers import (
    LoRALinearLayer_conditional, 
    LoRAConv2dLayer_conditional, 
    ConditionalLoRACompatibleLinear, 
    ConditionalLoRACompatibleConv
)
from .gnn import SparsePixelGraphEncoder
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config


class UNet2DConditionModelConditionalLora(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,)

    def add_lora_layers(
        self, 
        lora_rank: int, 
        tasks: list, 
        input_dims: list, 
        network_alpha: float = 12, 
        conditional_embedding_dim: int = 8
    ):
        self.lora_rank = lora_rank
        self.tasks = tasks
        self.n_tasks = len(tasks)
        original_conv_weights = self.conv_in.weight.data.clone()
        
        self.convs_in = nn.ModuleDict({})
        for input_dim, task in zip(input_dims, tasks):
            conv_in = torch.nn.Conv2d(
                4 + input_dim, 320, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            )
            with torch.no_grad():
                conv_in.weight[:, :4, :, :] = original_conv_weights
                conv_in.weight[:, 4:, :, :].normal_(0, 0.02)
            self.convs_in.update({task: conv_in})

        self.loras_dict = nn.ModuleDict({})
        linear_modules = [m for n, m in self.named_modules() if isinstance(m, nn.Linear)]

        loras = nn.ModuleList([])
        for m in linear_modules:
            lora_layer = LoRALinearLayer_conditional(
                m.in_features, m.out_features, 
                rank=lora_rank,
                network_alpha=network_alpha,
                embedding_dim=conditional_embedding_dim,
                num_modalities=self.n_tasks
            )
            loras.append(lora_layer)
        self.loras_dict.update({"linear_layers": loras})

        conv_modules = [m for n, m in self.named_modules() 
                       if isinstance(m, nn.Conv2d) and "conv_in" not in n and "convs_in" not in n]
        loras_conv = nn.ModuleList([])
        for m in conv_modules:
            lora_layer = LoRAConv2dLayer_conditional(
                m.in_channels, m.out_channels, 
                rank=lora_rank,
                kernel_size=m.kernel_size, 
                padding=m.padding,
                stride=m.stride,
                network_alpha=network_alpha,
                embedding_dim=conditional_embedding_dim,
                num_modalities=self.n_tasks
            )
            loras_conv.append(lora_layer)
        self.loras_dict.update({"conv_layers": loras_conv})

        self._replace_modules_with_lora_compatible()
    
    def _replace_module(self, module_path: str, new_module: nn.Module, old_module: nn.Module):
        new_module.weight.data.copy_(old_module.weight.data)
        if hasattr(old_module, 'bias') and old_module.bias is not None:
            new_module.bias.data.copy_(old_module.bias.data)
        else:
            new_module.bias = None
        
        parent = self
        *path, name = module_path.split('.')
        for part in path:
            parent = parent.get_submodule(part)
        parent._modules[name] = new_module
    
    def _replace_modules_with_lora_compatible(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'loras_dict' not in n:
                lora_linear = ConditionalLoRACompatibleLinear(m.in_features, m.out_features)
                self._replace_module(n, lora_linear, m)

            elif isinstance(m, nn.Conv2d) and 'loras_dict' not in n and "conv_in" not in n and "convs_in" not in n:
                lora_conv = ConditionalLoRACompatibleConv(
                    m.in_channels, m.out_channels,
                    kernel_size=m.kernel_size, 
                    padding=m.padding,
                    stride=m.stride
                )
                self._replace_module(n, lora_conv, m)
        
    def add_sparse_pixel_graph_encoder(
        self, 
        graph_dim: int = 64, 
        latent_dim: int = 1024, 
        num_heads: int = 4
    ):
        self.sparse_pixel_graph_encoder = SparsePixelGraphEncoder(
            graph_dim=graph_dim, 
            latent_dim=latent_dim, 
            num_heads=num_heads
        )

    def switch_lora(self, task: str):
        assert task in self.tasks, f"Task '{task}' not in {self.tasks}"
        self.conv_in = self.convs_in[task]

        idx_linear = 0
        idx_conv = 0
        modality_index = torch.LongTensor([self.tasks.index(task)])
        
        for n, m in self.named_modules():
            if isinstance(m, ConditionalLoRACompatibleLinear):
                lora = self.loras_dict["linear_layers"][idx_linear]
                lora.set_modality_index(modality_index)
                m.set_lora_layer(lora)
                idx_linear += 1
                
            if isinstance(m, ConditionalLoRACompatibleConv):
                lora = self.loras_dict["conv_layers"][idx_conv]
                lora.set_modality_index(modality_index)
                m.set_lora_layer(lora)
                idx_conv += 1

    def configure_optimizers(self, args):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.loras_dict.parameters():
            param.requires_grad = True
        for param in self.convs_in.parameters():
            param.requires_grad = True

        if hasattr(self, "sparse_pixel_graph_encoder"):
            for param in self.sparse_pixel_graph_encoder.parameters():
                param.requires_grad = True

        use_lora_plus = getattr(args, "lora_plus", False)

        if not use_lora_plus:
            params = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        # LoRA+: separate learning rates for down (A) and up (B) matrices
        lora_A_params = []
        lora_B_params = []
        base_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".down.weight" in name:
                lora_A_params.append(p)
            elif ".up.weight" in name:
                lora_B_params.append(p)
            else:
                base_params.append(p)

        lr_A = args.lora_lr_A if args.lora_lr_A is not None else args.learning_rate
        lr_B = args.lora_lr_B if args.lora_lr_B is not None else args.learning_rate
        wd_A = args.lora_wd_A if args.lora_wd_A is not None else args.adam_weight_decay
        wd_B = args.lora_wd_B if args.lora_wd_B is not None else args.adam_weight_decay

        optim_groups = []
        if base_params:
            optim_groups.append({"params": base_params, "lr": args.learning_rate, "weight_decay": args.adam_weight_decay})
        if lora_A_params:
            optim_groups.append({"params": lora_A_params, "lr": lr_A, "weight_decay": wd_A})
        if lora_B_params:
            optim_groups.append({"params": lora_B_params, "lr": lr_B, "weight_decay": wd_B})

        return torch.optim.AdamW(
            optim_groups,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
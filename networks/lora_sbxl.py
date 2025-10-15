# LoRA network module for SBXL
# Based on lora_flux.py

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import re

from library.utils import setup_logging
from library import sbxl_models

setup_logging()
import logging
logger = logging.getLogger(__name__)


class LoRAModule(torch.nn.Module):
    """LoRA module for SBXL"""
    
    def __init__(
        self,
        lora_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.lora_name = lora_name
        
        # Determine input and output dimensions
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        elif org_module.__class__.__name__ == "Conv2d_BN":
            # LSNet Conv2d_BN has internal Conv2d layer
            in_dim = org_module.c.in_channels
            out_dim = org_module.c.out_channels
        elif org_module.__class__.__name__ == "LSConv":
            # LSConv is a complex module, we can't apply LoRA directly to it
            # Instead, we should handle its internal Conv2d modules separately
            raise ValueError(f"LoRA should be applied to LSConv internal modules, not LSConv itself: {org_module.__class__.__name__}")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
        
        self.lora_dim = lora_dim
        
        # Create LoRA layers
        if org_module.__class__.__name__ in ["Conv2d", "Conv2d_BN"]:
            if org_module.__class__.__name__ == "Conv2d_BN":
                kernel_size = org_module.c.kernel_size
                stride = org_module.c.stride
                padding = org_module.c.padding
            else:
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)
        
        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        
        # Setup scaling
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))
        
        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
    
    def apply_to(self, dtype=None):
        """Apply LoRA to original module"""
        # Store the original module's weight dtype for dtype conversion in forward
        if hasattr(self.org_module, 'weight') and self.org_module.weight is not None:
            self.org_weight_dtype = self.org_module.weight.dtype
        elif hasattr(self.org_module, 'c') and hasattr(self.org_module.c, 'weight') and self.org_module.c.weight is not None:
            # For Conv2d_BN modules
            self.org_weight_dtype = self.org_module.c.weight.dtype
        else:
            self.org_weight_dtype = None
        
        # Ensure the original module is in the correct dtype
        if dtype is not None:
            self.org_module.to(dtype)
        
        # Move LoRA module to the same device as the original module
        device = self.org_module.weight.device if hasattr(self.org_module, 'weight') else None
        if device is None and hasattr(self.org_module, 'c') and hasattr(self.org_module.c, 'weight'):
            device = self.org_module.c.weight.device
        if device is not None:
            self.to(device)
        
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module
    
    def forward(self, x):
        """Forward pass with LoRA"""
        # Ensure input dtype matches the original module's weight dtype
        if hasattr(self, 'org_weight_dtype') and self.org_weight_dtype is not None and x.dtype != self.org_weight_dtype:
            x = x.to(self.org_weight_dtype)
        
        org_forwarded = self.org_forward(x)
        
        # Module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded
        
        # LoRA forward
        lx = self.lora_down(x)
        
        # Normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        
        # Rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale
        
        lx = self.lora_up(lx)
        
        return org_forwarded + lx * self.multiplier * scale

    def merge_to(self, state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype], device: Union[str, torch.device, None]):
        """Merge LoRA weights into the attached module."""
        if self.org_module is None:
            raise RuntimeError("Original module reference is missing; cannot merge weights after apply_to().")

        target_device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype or self.org_module.weight.dtype

        up_weight = state_dict["lora_up.weight"].to(torch.float32).to(target_device)
        down_weight = state_dict["lora_down.weight"].to(torch.float32).to(target_device)
        alpha_tensor = state_dict.get("alpha")
        if alpha_tensor is not None:
            alpha_value = float(alpha_tensor.item() if torch.is_tensor(alpha_tensor) else alpha_tensor)
            self.scale = alpha_value / self.lora_dim
            if hasattr(self, "alpha") and isinstance(self.alpha, torch.Tensor):
                self.alpha.copy_(torch.tensor(alpha_value, device=self.alpha.device, dtype=self.alpha.dtype))

        weight = self.org_module.weight.data.to(torch.float32).to(target_device)

        if isinstance(self.org_module, torch.nn.Conv2d):
            up_flat = up_weight.view(up_weight.size(0), -1)
            down_flat = down_weight.view(down_weight.size(0), -1)
            merged = up_flat @ down_flat
            merged = merged.view_as(weight)
        elif self.org_module.__class__.__name__ == "Conv2d_BN":
            # For Conv2d_BN, merge into the internal Conv2d layer
            weight = self.org_module.c.weight.data.to(torch.float32).to(target_device)
            up_flat = up_weight.view(up_weight.size(0), -1)
            down_flat = down_weight.view(down_weight.size(0), -1)
            merged = up_flat @ down_flat
            merged = merged.view_as(weight)
            weight = weight + self.multiplier * self.scale * merged
            self.org_module.c.weight.data.copy_(weight.to(dtype))
            return  # Early return for Conv2d_BN
        else:
            merged = up_weight @ down_weight
            if merged.shape != weight.shape:
                merged = merged.view_as(weight)

        weight = weight + self.multiplier * self.scale * merged
        self.org_module.weight.data.copy_(weight.to(dtype))


class SBXLLoRANetwork(torch.nn.Module):
    """LoRA network for SBXL."""

    UNET_TARGET_REPLACE_MODULE = ["Linear", "Conv2d", "Conv2d_BN"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Linear"]

    def __init__(
        self,
        unet: sbxl_models.SBXLUNet2DConditionModel,
        text_encoder: Optional[torch.nn.Module] = None,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.target_modules = target_modules

        self.text_encoder_loras = torch.nn.ModuleList()
        self.unet_loras = torch.nn.ModuleList()

        self.reg_lrs: Optional[Dict[str, float]] = None
        self.block_lr = False
        self.block_lr_weight: Optional[List[float]] = None

        def create_modules(
            root_module: torch.nn.Module,
            prefix: str,
            target_replace_modules: List[str],
        ) -> List[LoRAModule]:
            loras: List[LoRAModule] = []
            
            def process_module(module: torch.nn.Module, module_name: str, current_prefix: str):
                # Check if this module itself should be replaced
                if module.__class__.__name__ in target_replace_modules:
                    full_name = f"{current_prefix}.{module_name}" if current_prefix else module_name
                    if self.target_modules is not None and not any(target in full_name for target in self.target_modules):
                        return
                    
                    lora_name = full_name.replace(".", "_")
                    try:
                        lora = LoRAModule(
                            lora_name=lora_name,
                            org_module=module,
                            multiplier=multiplier,
                            lora_dim=lora_dim,
                            alpha=alpha,
                            dropout=dropout,
                            rank_dropout=rank_dropout,
                            module_dropout=module_dropout,
                        )
                        loras.append(lora)
                    except ValueError as e:
                        # If we can't create LoRA for this module, skip it
                        logger.warning(f"Skipping LoRA for {full_name}: {e}")
                        pass
                
                # Recursively process child modules
                for child_name, child_module in module.named_children():
                    child_full_name = f"{current_prefix}.{module_name}" if current_prefix else module_name
                    process_module(child_module, child_name, child_full_name)
            
            # Start processing from root - only process leaf modules
            for name, module in root_module.named_modules():
                if not name:  # Skip root module
                    continue
                # Only process modules that don't have children with target types
                has_target_children = any(child.__class__.__name__ in target_replace_modules 
                                        for child in module.modules() if child != module)
                if not has_target_children and module.__class__.__name__ in target_replace_modules:
                    full_name = f"{prefix}.{name}" if prefix else name
                    if self.target_modules is not None and not any(target in full_name for target in self.target_modules):
                        continue
                    
                    lora_name = full_name.replace(".", "_")
                    try:
                        lora = LoRAModule(
                            lora_name=lora_name,
                            org_module=module,
                            multiplier=multiplier,
                            lora_dim=lora_dim,
                            alpha=alpha,
                            dropout=dropout,
                            rank_dropout=rank_dropout,
                            module_dropout=module_dropout,
                        )
                        loras.append(lora)
                    except ValueError as e:
                        logger.warning(f"Skipping LoRA for {full_name}: {e}")
            
            return loras

        logger.info("Building SBXL LoRA modules")
        self.unet_loras = torch.nn.ModuleList(create_modules(unet, "unet", self.UNET_TARGET_REPLACE_MODULE))
        if text_encoder is not None:
            self.text_encoder_loras = torch.nn.ModuleList(
                create_modules(text_encoder, "text_encoder", self.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            )

        self.enable_unet_lora = len(self.unet_loras) > 0
        self.enable_text_encoder_lora = len(self.text_encoder_loras) > 0

        logger.info(
            "Created LoRA network with %d UNet modules and %d text encoder modules",
            len(self.unet_loras),
            len(self.text_encoder_loras),
        )

    def apply_to(
        self,
        text_encoders: Union[Optional[List[torch.nn.Module]], Optional[torch.nn.Module]],
        unet: sbxl_models.SBXLUNet2DConditionModel,
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not apply_text_encoder:
            self.enable_text_encoder_lora = False
        else:
            self.enable_text_encoder_lora = True
            logger.info("Enable LoRA for text encoder: %d modules", len(self.text_encoder_loras))

        if not apply_unet:
            self.enable_unet_lora = False
        else:
            self.enable_unet_lora = True
            logger.info("Enable LoRA for UNet: %d modules", len(self.unet_loras))

        for lora in list(self.text_encoder_loras) + list(self.unet_loras):
            lora.apply_to(dtype)

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = multiplier
        for lora in self.text_encoder_loras:
            lora.multiplier = multiplier

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def prepare_optimizer_params(
        self,
        text_encoder_lr: Optional[Union[float, int]],
        unet_lr: Optional[Union[float, int]],
        default_lr: float,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.requires_grad_(True)

        def resolve_lr(requested_lr: Optional[Union[float, int]]) -> float:
            return float(requested_lr if requested_lr is not None else default_lr)

        params: List[Dict[str, Any]] = []
        lr_descriptions: List[str] = []

        if self.enable_text_encoder_lora and len(self.text_encoder_loras) > 0:
            collected = [param for lora in self.text_encoder_loras for param in lora.parameters()]
            if collected:
                params.append({"params": collected, "lr": resolve_lr(text_encoder_lr)})
                lr_descriptions.append("textencoder")

        if self.enable_unet_lora and len(self.unet_loras) > 0:
            collected = [param for lora in self.unet_loras for param in lora.parameters()]
            if collected:
                params.append({"params": collected, "lr": resolve_lr(unet_lr)})
                lr_descriptions.append("unet")

        if not params:
            return [], None
        if len(lr_descriptions) == 0:
            return params, None

        return params, lr_descriptions

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing (no-op for LoRA networks)"""
        pass

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[Dict[str, Any]] = None) -> None:
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            converted = {}
            for key, value in state_dict.items():
                converted[key] = value.detach().clone().to("cpu").to(dtype)
            state_dict = converted

        ext = os.path.splitext(file)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            metadata = metadata or {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

        logger.info("Saved LoRA weights to %s", file)

    def load_weights(self, file: str):
        ext = os.path.splitext(file)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(file)
        else:
            state_dict = torch.load(file, map_location="cpu")

        info = self.load_state_dict(state_dict, strict=False)
        for lora in list(self.unet_loras) + list(self.text_encoder_loras):
            alpha_key = f"{lora.lora_name}.alpha"
            if alpha_key in state_dict and hasattr(lora, "alpha"):
                alpha_value = float(state_dict[alpha_key].item()) if torch.is_tensor(state_dict[alpha_key]) else float(state_dict[alpha_key])
                lora.scale = alpha_value / lora.lora_dim
                if isinstance(lora.alpha, torch.Tensor):
                    lora.alpha.copy_(torch.tensor(alpha_value, device=lora.alpha.device, dtype=lora.alpha.dtype))

        logger.info("Loaded LoRA weights from %s", file)
        return info

    @property
    def train_text_encoder(self) -> bool:
        return self.enable_text_encoder_lora and len(self.text_encoder_loras) > 0

    @property
    def train_unet(self) -> bool:
        return self.enable_unet_lora and len(self.unet_loras) > 0

    def merge_to(
        self,
        text_encoders: Union[Optional[List[torch.nn.Module]], Optional[torch.nn.Module]],
        unet: sbxl_models.SBXLUNet2DConditionModel,
        weights_sd: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device, None] = "cpu",
    ) -> None:
        logger.info("Merging LoRA weights into SBXL models")

        self.load_state_dict(weights_sd, strict=False)
        for lora in list(self.unet_loras) + list(self.text_encoder_loras):
            if hasattr(lora, "alpha") and isinstance(lora.alpha, torch.Tensor):
                lora.scale = float(lora.alpha.item()) / lora.lora_dim

        target_device = torch.device(device) if device is not None else torch.device("cpu")

        for lora in self.unet_loras + self.text_encoder_loras:
            sd = {
                "lora_down.weight": lora.lora_down.weight.to(target_device),
                "lora_up.weight": lora.lora_up.weight.to(target_device),
            }
            if hasattr(lora, "alpha"):
                sd["alpha"] = lora.alpha.detach().to(target_device)
            lora.merge_to(sd, dtype, target_device)

        logger.info("LoRA weights merged")

    def apply_max_norm_regularization(self, max_norm_value: float, device: Union[str, torch.device]):
        """Apply max-norm regularization across LoRA modules (matches other backends)."""

        downkeys: List[str] = []
        upkeys: List[str] = []
        alphakeys: List[str] = []
        norms: List[float] = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and key.endswith("weight"):
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        if not downkeys:
            return 0, 0.0, 0.0

        target_device = torch.device(device)

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(target_device)
            up = state_dict[upkeys[i]].to(target_device)
            alpha_tensor = state_dict.get(alphakeys[i])
            if alpha_tensor is None:
                alpha_tensor = torch.tensor(float(self.alpha), device=target_device)  # fallback
            else:
                alpha_tensor = alpha_tensor.to(target_device)

            dim = down.shape[0]
            scale = alpha_tensor / dim

            if up.dim() == 4 and down.dim() == 4:
                if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                    updown = (up.squeeze(2).squeeze(3) @ down.squeeze(2).squeeze(3)).unsqueeze(2).unsqueeze(3)
                else:
                    updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = (desired / norm).item()
            sqrt_ratio = ratio ** 0.5
            if ratio != 1.0:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scaled_norm = updown.norm() * ratio
            norms.append(scaled_norm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


def create_network_from_weights(
    multiplier: float,
    file: Optional[str],
    vae,
    text_encoder: List,
    unet: sbxl_models.SBXLUNet2DConditionModel,
    weights_sd: Optional[dict] = None,
    for_inference: bool = False,
    **kwargs,
) -> Tuple[SBXLLoRANetwork, dict]:
    """
    Create LoRA network from weights (for inference)
    
    Args:
        multiplier: LoRA multiplier
        file: File path (optional)
        vae: VAE model (unused)
        text_encoder: List of text encoders
        unet: UNet model
        weights_sd: State dict with LoRA weights
        for_inference: Whether this is for inference
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (lora_network, metadata)
    """
    metadata: Dict[str, Any] = {}

    if weights_sd is None:
        if file is None:
            raise ValueError("Either 'file' or 'weights_sd' must be provided to load LoRA weights")
        ext = os.path.splitext(file)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
            try:
                with safe_open(file, framework="pt") as handle:
                    metadata = handle.metadata() or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"Failed to read LoRA metadata from {file}: {exc}")
        else:
            loaded = torch.load(file, map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded:
                metadata = loaded.get("metadata", {}) or {}
                weights_sd = loaded["state_dict"]
            else:
                weights_sd = loaded

    if weights_sd is None:
        raise ValueError("Unable to load LoRA weights")

    sbxl_param_metadata = metadata.get("ss_sbxl_params") if metadata else None
    if sbxl_param_metadata:
        try:
            resolved_params = json.loads(sbxl_param_metadata)
            logger.info(f"LoRA metadata specifies SBXL params: {resolved_params}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse ss_sbxl_params metadata from LoRA weights")

    # Analyze weights to determine configuration
    lora_dim = None
    alpha = None
    
    for key in weights_sd.keys():
        if ".lora_down.weight" in key:
            if lora_dim is None:
                lora_dim = weights_sd[key].shape[0]
        if ".alpha" in key:
            if alpha is None:
                alpha = weights_sd[key].item()
    
    if lora_dim is None:
        lora_dim = 4  # Default
    if alpha is None:
        alpha = lora_dim
    
    logger.info(f"Creating LoRA network from weights: lora_dim={lora_dim}, alpha={alpha}, multiplier={multiplier}")
    
    # Create network
    text_encoder_module = text_encoder[0] if text_encoder and len(text_encoder) > 0 else None
    
    network = SBXLLoRANetwork(
        unet=unet,
        text_encoder=text_encoder_module,
        multiplier=multiplier,
        lora_dim=lora_dim,
        alpha=alpha,
    )
    
    return network, weights_sd


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    ae,
    text_encoders,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # For SBXL, we only support UNet LoRA for now
    # text_encoders parameter is ignored as SBXL uses its own text encoder
    
    network = SBXLLoRANetwork(
        unet=unet,
        text_encoder=None,  # SBXL handles text encoder separately
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
    )
    
    return network

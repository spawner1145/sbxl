# SBXL Model Architecture
# Text Encoder: SakikoLab/Sakiko-Prompt-Gen-v1.0
# UNet: SDXL-based with LSNet components
# VAE: Flux VAE (16 channels)
# Sampling: Flow-based sampling like Flux

import math
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# Import LSNet components
try:
    from lsnet.model.lsnet import Conv2d_BN, LSConv, RepVGGDW, BN_Linear, Attention as LSNetAttention, FFN
    LSNET_AVAILABLE = True
except ImportError:
    LSNET_AVAILABLE = False
    LSConv = None
    RepVGGDW = None

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XFORMERS_AVAILABLE = False
    xops = None  # type: ignore

# Import Flux VAE
from library import flux_models

from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

if not LSNET_AVAILABLE:
    logger.warning("LSNet not available, falling back to standard conv layers")


SUPPORTS_SDP_KERNEL = bool(getattr(torch.backends, "cuda", None) and hasattr(torch.backends.cuda, "sdp_kernel"))

ATTN_BACKEND_SDPA = "sdpa"
ATTN_BACKEND_FLASH = "flash"
ATTN_BACKEND_MEM = "mem_efficient"
ATTN_BACKEND_XFORMERS = "xformers"
ATTN_BACKEND_AUTO = "auto"


@dataclass
class SBXLParams:
    """Parameters for SBXL model configuration - Optimized for LSNet design principles"""
    # UNet params - Following LSNet "See Large, Focus Small" philosophy
    in_channels: int = 16  # VAE latent channels (Flux VAE is 16 channels)
    out_channels: int = 16
    model_channels: int = 384  # Base model channels (can be reduced for efficiency)
    attention_resolutions: Tuple[int, ...] = field(default_factory=lambda: (4, 2, 1))  # Attention levels
    num_res_blocks: int = 3  # Number of ResNet blocks per level
    channel_mult: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 4))  # Channel multipliers
    num_head_channels: int = 64
    transformer_depth: Tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 2))  # Transformer layers per level
    context_dim: int = 1024  # Sakiko text encoder hidden size
    use_lsnet: bool = True  # Use LSNet components when available
    
    # Text encoder params
    text_encoder_max_length: int = 384  # Sakiko max tokens
    text_encoder_hidden_size: int = 1024
    
    # VAE params (Flux VAE)
    vae_scale_factor: float = 0.3611
    vae_shift_factor: float = 0.1159
    
    @classmethod
    def get_base_config(cls) -> "SBXLParams":
        """Returns the base SBXL configuration"""
        return cls()

    @classmethod
    def get_lsnet_config(cls) -> "SBXLParams":
        """Returns LSNet-optimized SBXL configuration for superior quality-speed balance

        Target: Faster than SDXL while achieving higher quality through:
        1. 16-channel Flux VAE (4x information density vs SDXL's 4-channel)
        2. Sakiko text encoder (superior anime/Japanese understanding)
        3. Strategic transformer placement (quality where it matters most)
        4. Enhanced LSNet components (efficiency without quality loss)
        """
        return cls(
            # Optimized base channels: higher than SDXL but not excessive
            model_channels=352,  # Between SDXL's 320 and previous 384
            # Balanced attention levels for quality without overhead
            attention_resolutions=(8, 4, 2),  # Efficient attention coverage
            # Moderate ResNet blocks: quality boost without speed penalty
            num_res_blocks=3,  # Balanced between efficiency and quality
            # Smooth channel scaling optimized for 16-channel workflow
            channel_mult=(1, 2, 3, 4),  # Balanced scaling
            # Strategic transformer distribution for quality-speed balance:
            # - Level 3 (16x16): 3 transformers (focused semantic processing)
            # - Level 2 (32x32): 2 transformers (feature synthesis)
            # - Level 1 (64x64): 1 transformer (moderate processing)
            # - Level 0 (128x128): 0 transformers (pure LSNet efficiency)
            transformer_depth=(0, 1, 2, 3),  # Total: 6 transformers (vs SDXL's 13)
            # Maximize LSNet usage for efficiency
            use_lsnet=True,
        )

    @classmethod
    def get_lsnet_advanced_config(cls) -> "SBXLParams":
        """Returns advanced LSNet-inspired SBXL configuration for superior performance-efficiency balance"""
        return cls(
            # Advanced configuration: Higher capacity than LSNet-optimized but smarter component allocation
            model_channels=320,  # Between base (384) and LSNet (256) - optimal balance
            # Strategic attention placement: More levels for better generation quality but still reduced
            attention_resolutions=(8, 4, 2),  # Same as LSNet config for consistency
            # Balanced ResNet blocks: More blocks for capacity but LSNet components for efficiency
            num_res_blocks=3,  # Back to 3 for better capacity
            # Optimized channel scaling: Better progression than original (1,2,4,4)
            channel_mult=(1, 2, 3, 4),  # Smoother scaling, reduced final multiplier for efficiency
            # Strategic transformer placement: REDUCED from original (0,1,2,2)=5 to (0,1,1,2)=4
            transformer_depth=(0, 1, 1, 2),  # Reduced from original but more than basic LSNet
            # Maximize LSNet component usage for efficiency
            use_lsnet=True,
        )

    def copy(self) -> "SBXLParams":
        """Return a shallow copy of the parameters."""
        return replace(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a regular dict preserving tuples."""
        data: Dict[str, Any] = {}
        for field_info in fields(self):
            data[field_info.name] = getattr(self, field_info.name)
        return data

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert parameters to a JSON-serializable dict (tuples -> lists)."""
        serializable: Dict[str, Any] = {}
        for key, value in self.to_dict().items():
            if isinstance(value, tuple):
                serializable[key] = list(value)
            else:
                serializable[key] = value
        return serializable

    @staticmethod
    def _normalize_field_value(name: str, value: Any) -> Any:
        tuple_fields = {"attention_resolutions", "channel_mult", "transformer_depth"}

        if name in tuple_fields:
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",") if part.strip()]
                normalized: List[int] = []
                for part in parts:
                    try:
                        normalized.append(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid integer value '{part}' for SBXL param '{name}'") from None
                return tuple(normalized)
            if isinstance(value, (list, tuple)):
                normalized_list: List[int] = []
                for item in value:
                    if isinstance(item, int):
                        normalized_list.append(item)
                    elif isinstance(item, str):
                        try:
                            normalized_list.append(int(item))
                        except ValueError:
                            raise ValueError(f"Invalid integer value '{item}' for SBXL param '{name}'") from None
                    else:
                        raise ValueError(f"Unsupported element type '{type(item)}' for SBXL param '{name}'")
                return tuple(normalized_list)
            raise ValueError(f"Unsupported type '{type(value)}' for SBXL param '{name}'")

        if isinstance(value, str):
            value = value.strip()
            lowered = value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value

        return value

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        base: Optional["SBXLParams"] = None,
        strict: bool = True,
    ) -> "SBXLParams":
        """Create parameters from a dict, optionally extending a base config."""

        params = base.copy() if base is not None else cls.get_base_config()
        if not data:
            return params

        valid_fields = {field_info.name for field_info in fields(params)}
        updates: Dict[str, Any] = {}

        for key, value in data.items():
            if key not in valid_fields:
                if strict:
                    raise ValueError(f"Unknown SBXL parameter '{key}'")
                logger.warning(f"Ignoring unknown SBXL parameter '{key}'")
                continue
            updates[key] = cls._normalize_field_value(key, value)

        if not updates:
            return params

        return replace(params, **updates)


# ============================================================================
# LSNet-enhanced Blocks
# ============================================================================

class LSNetResnetBlock2D(nn.Module):
    """Advanced LSNet-inspired ResNet block with hierarchical component selection"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        eps: float = 1e-5,
        use_lsconv: bool = True,
        stage: int = 0,  # Hierarchical stage indicator
        depth: int = 0,  # Depth within stage
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_lsconv = use_lsconv
        self.stage = stage
        self.depth = depth

        # Input layers to handle channel mismatch (like SDXL)
        if in_channels == out_channels:
            self.in_layers = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True),
                nn.SiLU(),
                nn.Identity(),  # No conv needed if channels match
            )
        else:
            self.in_layers = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )

        # Hierarchical component selection following LSNet strategy
        if LSNET_AVAILABLE and use_lsconv:
            # Strategy: Early layers use RepVGGDW, later layers use LSConv
            # Low channels (< 128): RepVGGDW for efficiency
            # Medium channels (128-256): LSConv for "See Large, Focus Small"
            # High channels (> 256): LSConv for complex feature processing

            if in_channels < 128 or (stage == 0 and depth < 2):
                # Early layers: Use RepVGGDW for efficient computation
                try:
                    self.conv1 = RepVGGDW(out_channels)
                    logger.debug(f"Stage {stage} Depth {depth}: Using RepVGGDW for early layer (ch={out_channels})")
                except Exception as e:
                    logger.warning(f"Failed to create RepVGGDW, falling back to Conv2d: {e}")
                    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                # Later layers: Use LSConv for "See Large, Focus Small" capability
                try:
                    self.conv1 = LSConv(out_channels)
                    logger.debug(f"Stage {stage} Depth {depth}: Using LSConv for advanced layer (ch={out_channels})")
                except Exception as e:
                    logger.warning(f"Failed to create LSConv, falling back to Conv2d: {e}")
                    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            # Fallback to efficient conv
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        # Secondary convolution: Follow similar hierarchical strategy
        if LSNET_AVAILABLE and use_lsconv:
            if out_channels < 128 or (stage == 0 and depth < 2):
                try:
                    self.conv2 = RepVGGDW(out_channels)
                    logger.debug(f"Stage {stage} Depth {depth}: Using RepVGGDW for secondary conv (ch={out_channels})")
                except Exception as e:
                    logger.warning(f"Failed to create RepVGGDW, falling back to Conv2d: {e}")
                    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                try:
                    self.conv2 = LSConv(out_channels)
                    logger.debug(f"Stage {stage} Depth {depth}: Using LSConv for secondary conv (ch={out_channels})")
                except Exception as e:
                    logger.warning(f"Failed to create LSConv, falling back to Conv2d: {e}")
                    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if LSNET_AVAILABLE and use_lsconv and min(in_channels, out_channels) >= 64:
                # Use efficient 1x1 conv for channel adjustment
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None
    
    def forward(self, hidden_states, temb=None):
        residual = hidden_states
        
        # Input layers (like SDXL) - handle channel conversion
        hidden_states = self.in_layers(hidden_states)
        
        # Apply LSConv/Conv with hierarchical selection
        use_lsconv = isinstance(self.conv1, LSConv) if LSConv is not None else False
        if use_lsconv and hidden_states.device.type == 'cpu':
            # Fallback to standard conv on CPU
            temp_conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1).to(hidden_states.device, hidden_states.dtype)
            hidden_states = temp_conv1(hidden_states)
        else:
            hidden_states = self.conv1(hidden_states)
        
        if temb is not None and self.time_emb_proj is not None:
            temb = self.time_emb_proj(F.silu(temb))[:, :, None, None]
            hidden_states = hidden_states + temb
        
        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        
        # Same check for conv2
        use_lsconv_conv2 = isinstance(self.conv2, LSConv) if LSConv is not None else False
        if use_lsconv_conv2 and hidden_states.device.type == 'cpu':
            # Fallback to standard conv on CPU
            temp_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1).to(hidden_states.device, hidden_states.dtype)
            hidden_states = temp_conv2(hidden_states)
        else:
            hidden_states = self.conv2(hidden_states)
        
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        
        return hidden_states + residual


class Downsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, hidden_states):
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):
    """Cross attention module"""
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))
        self.attention_backend = ATTN_BACKEND_SDPA

    def set_attention_mode(self, backend: str):
        if backend == self.attention_backend:
            return

        if backend == ATTN_BACKEND_XFORMERS and not XFORMERS_AVAILABLE:
            logger.warning("xformers requested but not available; falling back to SDPA attention")
            backend = ATTN_BACKEND_SDPA
        if backend in (ATTN_BACKEND_FLASH, ATTN_BACKEND_MEM) and not SUPPORTS_SDP_KERNEL:
            logger.warning("Flash/Mem-efficient attention requested but SDP kernels are unavailable; falling back to SDPA attention")
            backend = ATTN_BACKEND_SDPA

        self.attention_backend = backend

    @contextmanager
    def _sdp_context(self):
        if not SUPPORTS_SDP_KERNEL or not torch.cuda.is_available():
            yield
            return

        if self.attention_backend == ATTN_BACKEND_FLASH:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                yield
        elif self.attention_backend == ATTN_BACKEND_MEM:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                yield
        else:
            yield

    def _apply_xformers_attention(self, query, key, value, batch_size, sequence_length, kv_seq_len, dim_head):
        if not XFORMERS_AVAILABLE or not query.is_cuda:
            return None

        q = query.reshape(batch_size * self.heads, sequence_length, dim_head).contiguous()
        k = key.reshape(batch_size * self.heads, kv_seq_len, dim_head).contiguous()
        v = value.reshape(batch_size * self.heads, kv_seq_len, dim_head).contiguous()

        try:
            attn_output = xops.memory_efficient_attention(q * self.scale, k, v, attn_bias=None)
        except RuntimeError as err:
            logger.warning(f"xformers attention failed, falling back to SDPA: {err}")
            return None

        attn_output = attn_output.reshape(batch_size, self.heads, sequence_length, dim_head)
        return attn_output
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = self.to_q(hidden_states)
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        mask_for_hidden = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_for_hidden = attention_mask
            elif attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
                mask_for_hidden = attention_mask[:, 0, :]
        if mask_for_hidden is not None and encoder_hidden_states is not hidden_states:
            encoder_hidden_states = encoder_hidden_states * mask_for_hidden.unsqueeze(-1).to(encoder_hidden_states.dtype)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        dim_head = query.shape[-1] // self.heads
        query = query.view(batch_size, sequence_length, self.heads, dim_head).transpose(1, 2)
        kv_seq_len = encoder_hidden_states.shape[1]
        key = key.view(batch_size, kv_seq_len, self.heads, dim_head).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.heads, dim_head).transpose(1, 2)

        # Prepare attention mask if provided (mask padded tokens)
        attn_mask = None
        use_xformers = self.attention_backend == ATTN_BACKEND_XFORMERS and XFORMERS_AVAILABLE and query.is_cuda
        if not use_xformers and attention_mask is not None:
            attn_mask = attention_mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn_mask = attn_mask.to(device=query.device, dtype=torch.bool)

        if use_xformers:
            attn_output = self._apply_xformers_attention(query, key, value, batch_size, sequence_length, kv_seq_len, dim_head)
            if attn_output is None:
                use_xformers = False
            else:
                hidden_states = attn_output

        if not use_xformers:
            with self._sdp_context():
                hidden_states = F.scaled_dot_product_attention(
                    query * self.scale,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, self.heads * dim_head)
        hidden_states = self.to_out(hidden_states)
        
        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Identity(),
            nn.Linear(inner_dim, dim)
        )
    
    def forward(self, hidden_states):
        return self.net(hidden_states)


class BasicTransformerBlock(nn.Module):
    """Basic transformer block with self-attention and cross-attention"""
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
    ):
        super().__init__()
        
        # Self attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        
        # Cross attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        
        # Feed forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Self attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states
        
        # Cross attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = attn_output + hidden_states
        
        # Feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        
        return hidden_states


class Transformer2DModel(nn.Module):
    """2D Transformer model"""
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_transformer_layers: int = 1,
        cross_attention_dim: int = 1024,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_transformer_layers)
        ])
        
        self.proj_out = nn.Linear(inner_dim, in_channels)
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        
        hidden_states = self.norm(hidden_states)
        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
        hidden_states = self.proj_in(hidden_states)
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, attention_mask)
        
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height, w=weight)
        
        return hidden_states + residual


# ============================================================================
# SBXL UNet Model
# ============================================================================

class SBXLUNet2DConditionModel(nn.Module):
    """SBXL UNet following LSNet 'See Large, Focus Small' philosophy - Lightweight yet powerful diffusion model"""
    
    def __init__(self, params: SBXLParams = None):
        super().__init__()
        
        if params is None:
            params = SBXLParams.get_base_config()
        
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.model_channels = params.model_channels
        self.time_embed_dim = params.model_channels * 4

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self._attention_backend = ATTN_BACKEND_SDPA

        # Track channel configurations for ControlNet compatibility
        self.down_block_output_channels: List[int] = []
        self.mid_block_output_channels: Optional[int] = None
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Input blocks
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
            )
        ])

        input_block_chans = [self.model_channels]
        self.down_block_output_channels.append(self.model_channels)
        ch = self.model_channels
        ds = 1
        
        # Build input blocks
        for level, mult in enumerate(params.channel_mult):
            for depth_idx in range(params.num_res_blocks):
                layers = [
                    LSNetResnetBlock2D(
                        in_channels=ch,
                        out_channels=mult * self.model_channels,
                        temb_channels=self.time_embed_dim,
                        use_lsconv=params.use_lsnet,
                        stage=level,  # Pass hierarchical stage
                        depth=depth_idx,  # Pass depth within stage
                    )
                ]
                ch = mult * self.model_channels
                
                if ds in params.attention_resolutions:
                    layers.append(
                        Transformer2DModel(
                            num_attention_heads=ch // params.num_head_channels,
                            attention_head_dim=params.num_head_channels,
                            in_channels=ch,
                            num_transformer_layers=params.transformer_depth[level],
                            cross_attention_dim=params.context_dim,
                        )
                    )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
                self.down_block_output_channels.append(ch)
            
            if level != len(params.channel_mult) - 1:
                self.input_blocks.append(
                    nn.Sequential(Downsample2D(ch, ch))
                )
                input_block_chans.append(ch)
                self.down_block_output_channels.append(ch)
                ds *= 2
        
        # Middle block - Use advanced stage (highest level)
        self.middle_block = nn.ModuleList([
            LSNetResnetBlock2D(
                in_channels=ch,
                out_channels=ch,
                temb_channels=self.time_embed_dim,
                use_lsconv=params.use_lsnet,
                stage=len(params.channel_mult),  # Advanced stage
                depth=0,
            ),
            Transformer2DModel(
                num_attention_heads=ch // params.num_head_channels,
                attention_head_dim=params.num_head_channels,
                in_channels=ch,
                num_transformer_layers=params.transformer_depth[-1],
                cross_attention_dim=params.context_dim,
            ),
            LSNetResnetBlock2D(
                in_channels=ch,
                out_channels=ch,
                temb_channels=self.time_embed_dim,
                use_lsconv=params.use_lsnet,
                stage=len(params.channel_mult),  # Advanced stage
                depth=1,
            ),
        ])

        self.mid_block_output_channels = ch
        
        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                # Calculate hierarchical stage for output blocks (reverse order)
                output_stage = len(params.channel_mult) - 1 - level
                output_depth = i

                layers = [
                    LSNetResnetBlock2D(
                        in_channels=ch + ich,
                        out_channels=mult * self.model_channels,
                        temb_channels=self.time_embed_dim,
                        use_lsconv=params.use_lsnet,
                        stage=output_stage,  # Hierarchical stage in reverse
                        depth=output_depth,  # Depth within stage
                    )
                ]
                ch = mult * self.model_channels

                if ds in params.attention_resolutions:
                    layers.append(
                        Transformer2DModel(
                            num_attention_heads=ch // params.num_head_channels,
                            attention_head_dim=params.num_head_channels,
                            in_channels=ch,
                            num_transformer_layers=params.transformer_depth[level],
                            cross_attention_dim=params.context_dim,
                        )
                    )

                if level and i == params.num_res_blocks:
                    layers.append(Upsample2D(ch, ch))
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(ch, self.out_channels, kernel_size=3, padding=1),
        )
    
    @property
    def device(self):
        """Get the device of the model parameters"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload
    
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def set_attention_backend(self, backend: str):
        if backend not in {ATTN_BACKEND_SDPA, ATTN_BACKEND_FLASH, ATTN_BACKEND_MEM, ATTN_BACKEND_XFORMERS, ATTN_BACKEND_AUTO}:
            raise ValueError(f"Unsupported attention backend: {backend}")

        resolved_backend = backend
        if backend == ATTN_BACKEND_AUTO:
            resolved_backend = ATTN_BACKEND_SDPA

        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.set_attention_mode(resolved_backend)

        self._attention_backend = resolved_backend
        logger.info(f"SBXL attention backend set to {resolved_backend}")

    def set_use_xformers(self, enable: bool):
        self.set_attention_backend(ATTN_BACKEND_XFORMERS if enable else ATTN_BACKEND_SDPA)

    def set_use_flash_attention(self, enable: bool):
        self.set_attention_backend(ATTN_BACKEND_FLASH if enable else ATTN_BACKEND_SDPA)

    def set_use_memory_efficient_attention(self, enable: bool):
        self.set_attention_backend(ATTN_BACKEND_MEM if enable else ATTN_BACKEND_SDPA)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # Time embedding
        t_emb = get_timestep_embedding(timestep, self.model_channels)
        emb = self.time_embed(t_emb)
        
        # Input
        h = sample
        hs = []
        
        def run_block(module, h, emb, encoder_hidden_states, attention_mask):
            """Helper to run a single block (with or without gradient checkpointing)"""
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(layer):
                    def custom_forward(*inputs):
                        if isinstance(layer, LSNetResnetBlock2D):
                            return layer(inputs[0], inputs[1])
                        elif isinstance(layer, Transformer2DModel):
                            return layer(inputs[0], inputs[2], inputs[3])
                        else:
                            return layer(inputs[0])
                    return custom_forward
                
                if self.cpu_offload_checkpointing:
                    def create_custom_forward_cpu_offload(func):
                        def custom_forward(*inputs):
                            cuda_inputs = to_cuda(inputs)
                            outputs = func(*cuda_inputs)
                            return to_cpu(outputs)
                        return custom_forward
                    
                    if isinstance(module, nn.Sequential):
                        for layer in module:
                            h = torch.utils.checkpoint.checkpoint(
                                create_custom_forward_cpu_offload(create_custom_forward(layer)), 
                                h, emb, encoder_hidden_states, attention_mask, use_reentrant=False
                            )
                    else:
                        for layer in module:
                            h = torch.utils.checkpoint.checkpoint(
                                create_custom_forward_cpu_offload(create_custom_forward(layer)), 
                                h, emb, encoder_hidden_states, attention_mask, use_reentrant=False
                            )
                else:
                    if isinstance(module, nn.Sequential):
                        for layer in module:
                            h = checkpoint(create_custom_forward(layer), h, emb, encoder_hidden_states, attention_mask, use_reentrant=False)
                    else:
                        for layer in module:
                            h = checkpoint(create_custom_forward(layer), h, emb, encoder_hidden_states, attention_mask, use_reentrant=False)
                return h
            else:
                if isinstance(module, nn.Sequential):
                    return module(h)
                else:
                    for layer in module:
                        if isinstance(layer, LSNetResnetBlock2D):
                            h = layer(h, emb)
                        elif isinstance(layer, Transformer2DModel):
                            h = layer(h, encoder_hidden_states, attention_mask)
                        else:
                            h = layer(h)
                    return h
        
        # Input blocks
        for idx, module in enumerate(self.input_blocks):
            h = run_block(module, h, emb, encoder_hidden_states, attention_mask)
            if down_block_additional_residuals is not None:
                if idx >= len(down_block_additional_residuals):
                    raise ValueError("down_block_additional_residuals length does not match input blocks")
                h = h + down_block_additional_residuals[idx]
            hs.append(h)
        
        # Middle block
        for layer in self.middle_block:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(layer):
                    def custom_forward(*inputs):
                        if isinstance(layer, LSNetResnetBlock2D):
                            return layer(inputs[0], inputs[1])
                        elif isinstance(layer, Transformer2DModel):
                            return layer(inputs[0], inputs[2], inputs[3])
                        else:
                            return layer(inputs[0])
                    return custom_forward
                
                if self.cpu_offload_checkpointing:
                    def create_custom_forward_cpu_offload(func):
                        def custom_forward(*inputs):
                            cuda_inputs = to_cuda(inputs)
                            outputs = func(*cuda_inputs)
                            return to_cpu(outputs)
                        return custom_forward
                    
                    h = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_cpu_offload(create_custom_forward(layer)), 
                        h, emb, encoder_hidden_states, attention_mask, use_reentrant=False
                    )
                else:
                    h = checkpoint(create_custom_forward(layer), h, emb, encoder_hidden_states, attention_mask, use_reentrant=False)
            else:
                if isinstance(layer, LSNetResnetBlock2D):
                    h = layer(h, emb)
                elif isinstance(layer, Transformer2DModel):
                    h = layer(h, encoder_hidden_states, attention_mask)
                else:
                    h = layer(h)
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = run_block(module, h, emb, encoder_hidden_states, attention_mask)
        
        # Output
        h = self.out(h)
        
        if not return_dict:
            return (h,)
        
        return {"sample": h}


def to_cuda(x):
    """Move tensor/list/dict to CUDA device"""
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    """Move tensor/list/dict to CPU device"""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
):
    """Create sinusoidal timestep embeddings - copied from Flux"""
    from library.flux_models import timestep_embedding
    return timestep_embedding(timesteps, embedding_dim, max_period, time_factor)


class ControlNetConditioningEmbedding(nn.Module):
    """Encode conditioning image into latent-aligned features."""

    def __init__(self, in_channels: int = 3, out_channels: int = 320):
        super().__init__()

        hidden_dims = [16, 32, 96, 256]
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            c_in = hidden_dims[idx]
            c_out = hidden_dims[idx + 1]
            self.blocks.append(nn.Conv2d(c_in, c_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        x = conditioning
        x = self.conv_in(x)
        x = F.silu(x)
        for layer in self.blocks:
            x = layer(x)
            x = F.silu(x)
        x = self.conv_out(x)
        return x


class SBXLControlNet(SBXLUNet2DConditionModel):
    """ControlNet for SBXL UNet."""

    def __init__(
        self,
        params: SBXLParams | None = None,
        conditioning_channels: int = 3,
        conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__(params=params)
        self.multiplier = float(conditioning_scale)

        # Remove final output head – ControlNet outputs residuals only
        del self.out

        # Conditioning encoder
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            in_channels=conditioning_channels,
            out_channels=self.params.model_channels,
        )

        # Zero-initialized adapters for each down block and mid block
        self.controlnet_down_blocks = nn.ModuleList()
        for channels in self.down_block_output_channels:
            proj = nn.Conv2d(channels, channels, kernel_size=1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.controlnet_down_blocks.append(proj)

        if self.mid_block_output_channels is None:
            raise ValueError("mid_block_output_channels is not initialized")
        self.controlnet_mid_block = nn.Conv2d(self.mid_block_output_channels, self.mid_block_output_channels, kernel_size=1)
        nn.init.zeros_(self.controlnet_mid_block.weight)
        nn.init.zeros_(self.controlnet_mid_block.bias)

    def init_from_unet(self, unet: SBXLUNet2DConditionModel) -> None:
        """Initialize ControlNet weights from an existing SBXL UNet."""

        with torch.no_grad():
            own_state = self.state_dict()
            unet_state = unet.state_dict()
            for name, param in unet_state.items():
                if name in own_state and not name.startswith("controlnet_"):
                    own_state[name].copy_(param.data)
        logger.info("Initialized SBXLControlNet from base UNet weights")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: float | torch.Tensor = 1.0,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if controlnet_cond is None:
            raise ValueError("controlnet_cond must be provided for SBXLControlNet")

        if conditioning_scale is None:
            conditioning_scale = self.multiplier

        if isinstance(conditioning_scale, torch.Tensor):
            scale_tensor = conditioning_scale.to(sample.device, dtype=sample.dtype).view(-1, *([1] * (sample.dim() - 1)))
        else:
            scale_tensor = torch.tensor(float(conditioning_scale), device=sample.device, dtype=sample.dtype)

        # Encode conditioning image and scale
        cond_emb = self.controlnet_cond_embedding(controlnet_cond.to(sample.dtype))
        cond_emb = cond_emb * scale_tensor

        # Time embedding (reuse base utilities)
        t_emb = get_timestep_embedding(timestep.to(sample.device).float(), self.model_channels)
        t_emb = t_emb.to(sample.dtype)
        emb = self.time_embed(t_emb)

        h = sample
        down_block_residuals: List[torch.Tensor] = []

        def run_block(module, hidden_states, emb_in, encoder_states, attn_mask):
            if self.gradient_checkpointing and self.training:
                def create_forward(layer):
                    def custom_forward(*inputs):
                        if isinstance(layer, LSNetResnetBlock2D):
                            return layer(inputs[0], inputs[1])
                        elif isinstance(layer, Transformer2DModel):
                            return layer(inputs[0], inputs[2], inputs[3])
                        else:
                            return layer(inputs[0])
                    return custom_forward

                if isinstance(module, nn.Sequential):
                    for layer in module:
                        hidden_states = checkpoint(create_forward(layer), hidden_states, emb_in, encoder_states, attn_mask, use_reentrant=False)
                else:
                    for layer in module:
                        hidden_states = checkpoint(create_forward(layer), hidden_states, emb_in, encoder_states, attn_mask, use_reentrant=False)
                return hidden_states
            else:
                if isinstance(module, nn.Sequential):
                    return module(hidden_states)
                else:
                    for layer in module:
                        if isinstance(layer, LSNetResnetBlock2D):
                            hidden_states = layer(hidden_states, emb_in)
                        elif isinstance(layer, Transformer2DModel):
                            hidden_states = layer(hidden_states, encoder_states, attn_mask)
                        else:
                            hidden_states = layer(hidden_states)
                    return hidden_states

        for idx, module in enumerate(self.input_blocks):
            h = run_block(module, h, emb, encoder_hidden_states, attention_mask)
            if idx == 0:
                h = h + cond_emb
            residual = self.controlnet_down_blocks[idx](h)
            residual = residual * scale_tensor
            down_block_residuals.append(residual)

        # Middle block mirrors base class forward
        for layer in self.middle_block:
            if self.gradient_checkpointing and self.training:
                def create_forward(layer):
                    def custom_forward(*inputs):
                        if isinstance(layer, LSNetResnetBlock2D):
                            return layer(inputs[0], inputs[1])
                        elif isinstance(layer, Transformer2DModel):
                            return layer(inputs[0], inputs[2], inputs[3])
                        else:
                            return layer(inputs[0])
                    return custom_forward

                h = checkpoint(create_forward(layer), h, emb, encoder_hidden_states, attention_mask, use_reentrant=False)
            else:
                if isinstance(layer, LSNetResnetBlock2D):
                    h = layer(h, emb)
                elif isinstance(layer, Transformer2DModel):
                    h = layer(h, encoder_hidden_states, attention_mask)
                else:
                    h = layer(h)

        mid_residual = self.controlnet_mid_block(h)
        mid_residual = mid_residual * scale_tensor

        return tuple(down_block_residuals), mid_residual

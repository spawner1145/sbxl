# SBXL Utility Functions
# Model loading, saving, and utility functions

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModel

try:  # TOML is optional but available in training environments
    import toml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    toml = None

from library.utils import setup_logging
from library.safetensors_utils import load_safetensors
from library import sbxl_models, flux_models

setup_logging()
import logging
logger = logging.getLogger(__name__)


MODEL_VERSION_SBXL_V1 = "sbxl1"
SAKIKO_TEXT_ENCODER_ID = "SakikoLab/Sakiko-Prompt-Gen-v1.0"


def _coerce_cli_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()

    if lowered in {"true", "false"}:
        return lowered == "true"

    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse SBXL parameter JSON literal: {value}")

    try:
        return int(value, 0)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def parse_sbxl_param_overrides(param_pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Parse CLI-provided key=value overrides for SBXL parameters."""

    if not param_pairs:
        return {}

    overrides: Dict[str, Any] = {}
    for pair in param_pairs:
        if "=" not in pair:
            raise ValueError(f"SBXL parameter override must use key=value format: '{pair}'")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"SBXL parameter override is missing a key: '{pair}'")
        overrides[key] = _coerce_cli_value(raw_value)

    return overrides


def load_sbxl_params_file(file_path: str) -> Dict[str, Any]:
    """Load SBXL parameter overrides from a JSON or TOML file."""

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SBXL params file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".json", ".jsn"}:
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    elif ext in {".toml", ".tml"}:
        if toml is None:
            raise ImportError("toml package is required to read TOML SBXL params files")
        data = toml.load(file_path)
    else:
        raise ValueError(f"Unsupported SBXL params file extension '{ext}' (expected JSON or TOML)")

    if not isinstance(data, dict):
        raise ValueError("SBXL params configuration must be a dictionary at the top level")

    return data


def get_sbxl_params_from_args(
    params_file: Optional[str],
    param_pairs: Optional[List[str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Resolve SBXL parameter configuration supplied via CLI arguments."""

    base_overrides: Optional[Dict[str, Any]] = None
    overrides: Optional[Dict[str, Any]] = None

    if params_file:
        file_overrides = load_sbxl_params_file(params_file)
        try:
            validated_params = sbxl_models.SBXLParams.from_dict(file_overrides, strict=False)
        except Exception as exc:
            raise ValueError(f"Failed to apply SBXL params from file '{params_file}': {exc}") from exc

        base_config = sbxl_models.SBXLParams.get_base_config()
        diff: Dict[str, Any] = {}
        for field_name, value in validated_params.to_dict().items():
            if getattr(base_config, field_name) != value:
                diff[field_name] = value

        if diff:
            base_overrides = diff
            logger.info(f"Loaded SBXL parameter preset from {params_file}")
        else:
            logger.info(f"Loaded SBXL parameter preset from {params_file} (no differences from base config)")

    parsed_overrides = parse_sbxl_param_overrides(param_pairs)
    if parsed_overrides:
        try:
            validated_overrides = sbxl_models.SBXLParams.from_dict(parsed_overrides, strict=True)
        except Exception as exc:
            raise ValueError(f"Invalid SBXL parameter override: {exc}") from exc

        sanitized: Dict[str, Any] = {}
        for key in parsed_overrides.keys():
            sanitized[key] = getattr(validated_overrides, key)
        overrides = sanitized

    return base_overrides, overrides


def _load_sbxl_params_from_metadata(ckpt_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return None

    try:
        with safe_open(ckpt_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"Failed to read SBXL metadata from {ckpt_path}: {exc}")
        return None

    raw_params = metadata.get("sbxl_params") if metadata is not None else None
    if not raw_params:
        return None

    try:
        params_dict = json.loads(raw_params)
        if not isinstance(params_dict, dict):
            raise ValueError("sbxl_params metadata must be a JSON object")
        return params_dict
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to parse sbxl_params metadata from {ckpt_path}: {exc}")
        return None


def resolve_sbxl_params(
    base_overrides: Optional[Dict[str, Any]],
    ckpt_path: Optional[str],
    overrides: Optional[Dict[str, Any]],
) -> sbxl_models.SBXLParams:
    """Derive the SBXLParams to use, honoring metadata and CLI overrides."""

    metadata_overrides = _load_sbxl_params_from_metadata(ckpt_path)

    params = sbxl_models.SBXLParams.get_base_config()

    if metadata_overrides:
        try:
            params = sbxl_models.SBXLParams.from_dict(metadata_overrides, params, strict=False)
            logger.info("Applied SBXL architecture parameters from checkpoint metadata")
        except Exception as exc:
            logger.warning(f"Failed to apply SBXL params from metadata: {exc}")

    if base_overrides:
        params = sbxl_models.SBXLParams.from_dict(base_overrides, params, strict=True)

    if overrides:
        params = sbxl_models.SBXLParams.from_dict(overrides, params, strict=True)
        logger.info("Applied SBXL architecture overrides from CLI")

    logger.debug("Resolved SBXL parameters: %s", params.to_serializable_dict())

    return params


def load_sbxl_unet(
    ckpt_path: Optional[str],
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    preset_overrides: Optional[Union[sbxl_models.SBXLParams, Dict[str, Any]]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> sbxl_models.SBXLUNet2DConditionModel:
    """
    Load SBXL UNet model from checkpoint
    
    Args:
        ckpt_path: Path to the checkpoint file
        dtype: Data type for model weights
        device: Device to load model to
        disable_mmap: Whether to disable memory mapping
    
    Returns:
        Loaded SBXL UNet model
    """
    logger.info("Building SBXL UNet model")

    preset_dict: Optional[Dict[str, Any]]
    if isinstance(preset_overrides, sbxl_models.SBXLParams):
        preset_dict = preset_overrides.to_dict()
    else:
        preset_dict = preset_overrides

    resolved_params = resolve_sbxl_params(preset_dict, ckpt_path, cli_overrides)

    # Build model directly on target device to avoid meta tensor issues
    model = sbxl_models.SBXLUNet2DConditionModel(resolved_params)
    if dtype is not None:
        model = model.to(dtype)
    
    # Load state dict
    if ckpt_path:
        if os.path.isfile(ckpt_path):
            logger.info(f"Loading state dict from {ckpt_path}")
            sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

            # Remove any annoying prefix
            for key in list(sd.keys()):
                new_key = key.replace("model.diffusion_model.", "")
                if new_key != key:
                    sd[new_key] = sd.pop(key)

            info = model.load_state_dict(sd, strict=False, assign=True)
            logger.info(f"Loaded SBXL UNet: {info}")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}, keeping random initialization")
    else:
        logger.info("No pretrained SBXL UNet provided; training will start from random initialization")
    
    # Model is already on target device
    return model


def save_sbxl_unet(
    model: sbxl_models.SBXLUNet2DConditionModel,
    save_path: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, Union[str, float, int]]] = None,
):
    """Save SBXL UNet weights to file.

    Args:
        model: model to persist
        save_path: destination path (extension determines format)
        dtype: optional dtype cast for tensors before saving
        metadata: optional metadata dictionary (values converted to string for safetensors)
    """

    logger.info(f"Saving SBXL UNet model to {save_path}")

    # Normalize state dict tensors to CPU and target dtype
    cast_dtype = dtype
    state_dict = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().to("cpu")
        if cast_dtype is not None and tensor.dtype != cast_dtype:
            tensor = tensor.to(cast_dtype)
        state_dict[key] = tensor

    combined_metadata: Dict[str, Union[str, float, int]] = dict(metadata or {})
    if hasattr(model, "params") and isinstance(model.params, sbxl_models.SBXLParams):
        combined_metadata.setdefault("sbxl_params", json.dumps(model.params.to_serializable_dict()))

    safetensors_metadata = {k: str(v) for k, v in combined_metadata.items()}

    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".safetensors":
        save_file(state_dict, save_path, metadata=safetensors_metadata)
    else:
        torch.save({"state_dict": state_dict, "metadata": combined_metadata}, save_path)

    logger.info(f"Saved SBXL UNet to {save_path}")


def load_text_encoder(
    text_encoder_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    cache_dir: Optional[str] = None,
) -> AutoModel:
    """
    Load Sakiko text encoder
    
    Args:
        text_encoder_path: Path to text encoder checkpoint (or HuggingFace model ID)
        dtype: Data type for model weights
        device: Device to load model to
        disable_mmap: Whether to disable memory mapping
        cache_dir: Cache directory for HuggingFace models
    
    Returns:
        Loaded text encoder model
    """
    if text_encoder_path is None:
        text_encoder_path = SAKIKO_TEXT_ENCODER_ID
    
    logger.info(f"Loading Sakiko text encoder from {text_encoder_path}")
    
    try:
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        if device is not None:
            text_encoder.to(device)
        current_dtype = getattr(text_encoder, "dtype", None)
        if dtype is not None and current_dtype is not None and current_dtype != dtype:
            text_encoder.to(dtype)

        logger.info(f"Loaded text encoder: {text_encoder.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load text encoder: {e}")
        raise

    return text_encoder


def load_tokenizer(
    tokenizer_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> AutoTokenizer:
    """
    Load Sakiko tokenizer
    
    Args:
        tokenizer_path: Path to tokenizer (or HuggingFace model ID)
        cache_dir: Cache directory for HuggingFace models
    
    Returns:
        Loaded tokenizer
    """
    if tokenizer_path is None:
        tokenizer_path = SAKIKO_TEXT_ENCODER_ID
    
    logger.info(f"Loading Sakiko tokenizer from {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"
        logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    return tokenizer


def load_vae(
    ckpt_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> flux_models.AutoEncoder:
    """
    Load the AutoEncoder model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (Union[str, torch.device]): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.

    Returns:
        ae (flux_models.AutoEncoder): The loaded model.
    """
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        # dev and schnell have the same AE params
        ae = flux_models.AutoEncoder(flux_models.configs["schnell"].ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=device, disable_mmap=disable_mmap, dtype=dtype)

    # Neta-Lumina support
    if "vae.decoder.conv_in.bias" in sd:
        # remove "vae." prefix
        filtered_sd = {k.replace("vae.", ""): v for k, v in sd.items() if k.startswith("vae.")}
        sd = filtered_sd

    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")

    return ae


def select_attention_backend_from_args(args) -> str:
    """Derive attention backend from command-line arguments."""

    if getattr(args, "use_flash_attn", False):
        return sbxl_models.ATTN_BACKEND_FLASH
    if getattr(args, "mem_eff_attn", False):
        return sbxl_models.ATTN_BACKEND_MEM
    if getattr(args, "xformers", False):
        return sbxl_models.ATTN_BACKEND_XFORMERS
    if getattr(args, "sdpa", False):
        return sbxl_models.ATTN_BACKEND_SDPA
    return sbxl_models.ATTN_BACKEND_AUTO


def configure_attention_backend(model: torch.nn.Module, backend: str):
    """Configure the attention backend for SBXL-compatible modules."""

    if backend == sbxl_models.ATTN_BACKEND_AUTO:
        return

    if not hasattr(model, "set_attention_backend"):
        logger.debug("Model does not support attention backend configuration; skipping")
        return

    try:
        model.set_attention_backend(backend)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to set attention backend to {backend}: {exc}")


def load_controlnet(
    controlnet_path: Optional[str],
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    preset_overrides: Optional[Union[sbxl_models.SBXLParams, Dict[str, Any]]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> sbxl_models.SBXLControlNet:
    """Load SBXL ControlNet model."""

    logger.info("Building SBXL ControlNet model")
    preset_dict: Optional[Dict[str, Any]]
    if isinstance(preset_overrides, sbxl_models.SBXLParams):
        preset_dict = preset_overrides.to_dict()
    else:
        preset_dict = preset_overrides

    resolved_params = resolve_sbxl_params(preset_dict, controlnet_path, cli_overrides)
    controlnet = sbxl_models.SBXLControlNet(resolved_params)
    if dtype is not None:
        controlnet = controlnet.to(dtype)

    if controlnet_path is not None and os.path.exists(controlnet_path):
        logger.info(f"Loading ControlNet state dict from {controlnet_path}")
        state_dict = load_safetensors(controlnet_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

        for key in list(state_dict.keys()):
            new_key = key.replace("model.diffusion_model.", "")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

        info = controlnet.load_state_dict(state_dict, strict=False, assign=True)
        logger.info(f"Loaded ControlNet weights: {info}")
    else:
        logger.info("ControlNet checkpoint not provided; using freshly initialized weights")

    controlnet.to(device=device, dtype=dtype if dtype is not None else controlnet.dtype)
    return controlnet


def save_controlnet(
    controlnet: sbxl_models.SBXLControlNet,
    save_path: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, Union[str, float, int]]] = None,
) -> None:
    """Save SBXL ControlNet weights to file.

    Args:
        controlnet: model to persist
        save_path: destination path (extension determines format)
        dtype: optional dtype cast for tensors before saving
        metadata: optional metadata dictionary (values converted to string for safetensors)
    """

    logger.info(f"Saving ControlNet to {save_path}")

    cast_dtype = dtype
    state_dict = {}
    for key, value in controlnet.state_dict().items():
        tensor = value.detach().to("cpu")
        if cast_dtype is not None and tensor.dtype != cast_dtype:
            tensor = tensor.to(cast_dtype)
        state_dict[key] = tensor

    combined_metadata: Dict[str, Union[str, float, int]] = dict(metadata or {})
    if hasattr(controlnet, "params") and isinstance(controlnet.params, sbxl_models.SBXLParams):
        combined_metadata.setdefault("sbxl_params", json.dumps(controlnet.params.to_serializable_dict()))

    safetensors_metadata = {k: str(v) for k, v in combined_metadata.items()}

    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".safetensors":
        save_file(state_dict, save_path, metadata=safetensors_metadata)
    else:
        torch.save({"state_dict": state_dict, "metadata": combined_metadata}, save_path)

    logger.info(f"Saved ControlNet to {save_path}")


def create_controlnet_from_unet(
    unet: sbxl_models.SBXLUNet2DConditionModel,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> sbxl_models.SBXLControlNet:
    """Create a ControlNet initialized from a base SBXL UNet."""

    params = getattr(unet, "params", sbxl_models.SBXLParams.get_base_config()).copy()
    controlnet = sbxl_models.SBXLControlNet(params)
    if dtype is not None:
        controlnet = controlnet.to(dtype)
    controlnet.init_from_unet(unet)
    backend = getattr(unet, "_attention_backend", sbxl_models.ATTN_BACKEND_SDPA)
    try:
        controlnet.set_attention_backend(backend)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to align ControlNet attention backend ({backend}): {exc}")
    controlnet.to(device=device, dtype=dtype if dtype is not None else controlnet.dtype)
    return controlnet


def prepare_img_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
    """
    Prepare image IDs for positional encoding (similar to Flux)
    
    Args:
        batch_size: Batch size
        height: Image height in latent space
        width: Image width in latent space
        device: Device to create tensor on
        dtype: Data type for tensor
    
    Returns:
        Image IDs tensor
    """
    img_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(height, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(width, device=device, dtype=dtype)[None, :]
    img_ids = img_ids.reshape(-1, 3)
    img_ids = img_ids[None, :, :].expand(batch_size, -1, -1)
    return img_ids


def get_noise(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate noise tensor for Flux VAE (16 channels)
    
    Args:
        batch_size: Batch size
        height: Height in latent space
        width: Width in latent space
        device: Device to create tensor on
        dtype: Data type for tensor
        seed: Random seed
    
    Returns:
        Noise tensor
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    noise = torch.randn(
        batch_size,
        16,  # Flux VAE latent channels (16ch)
        height,
        width,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    return noise


def apply_attention_mask(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Apply attention mask to hidden states
    
    Args:
        hidden_states: Hidden states tensor
        attention_mask: Attention mask tensor
    
    Returns:
        Masked hidden states
    """
    if attention_mask is not None:
        # Expand mask to match hidden states dimensions
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-1)
        hidden_states = hidden_states * attention_mask
    
    return hidden_states


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Unpack latents from sequence to spatial format
    
    Args:
        latents: Latents in sequence format [batch, seq_len, channels]
        height: Height in latent space
        width: Width in latent space
    
    Returns:
        Latents in spatial format [batch, channels, height, width]
    """
    batch_size = latents.shape[0]
    latents = latents.reshape(batch_size, height, width, -1)
    latents = latents.permute(0, 3, 1, 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack latents from spatial to sequence format
    
    Args:
        latents: Latents in spatial format [batch, channels, height, width]
    
    Returns:
        Latents in sequence format [batch, seq_len, channels]
    """
    batch_size, channels, height, width = latents.shape
    latents = latents.permute(0, 2, 3, 1)
    latents = latents.reshape(batch_size, height * width, channels)
    return latents


def encode_prompts(
    prompts: Union[str, list[str]],
    tokenizer: AutoTokenizer,
    text_encoder: AutoModel,
    max_length: int = 384,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompts using Sakiko text encoder with system prompt support
    
    Args:
        prompts: Prompt(s) to encode
        tokenizer: Tokenizer
        text_encoder: Text encoder model
        max_length: Maximum sequence length
        device: Device to create tensors on
        dtype: Data type for tensors
        system_prompt: System prompt to prepend
        use_chat_template: Whether to use chat template
    
    Returns:
        Tuple of (hidden_states, attention_mask)
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Apply system prompt
    if system_prompt is None:
        system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
    
    processed_prompts = []
    use_template = use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    for prompt in prompts:
        if use_template:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False
                )
            except Exception as e:
                logger.warning(f"chat_template failed, using text concatenation: {e}")
                full_prompt = f'{system_prompt} <Prompt Start> {prompt}'
        else:
            full_prompt = f'{system_prompt} <Prompt Start> {prompt}'
        processed_prompts.append(full_prompt)
    
    # Tokenize
    tokens = tokenizer(
        processed_prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # Ensure tensors live on same device as text encoder
    try:
        encoder_device = next(text_encoder.parameters()).device  # type: ignore[call-arg]
    except StopIteration:
        encoder_device = device

    input_ids = tokens["input_ids"].to(encoder_device)
    attention_mask = tokens["attention_mask"].to(encoder_device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    if outputs.hidden_states is not None and len(outputs.hidden_states) >= 2:
        hidden_states = outputs.hidden_states[-2]
    else:
        hidden_states = outputs.last_hidden_state

    hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    hidden_states = hidden_states.to(device)
    if dtype is not None:
        hidden_states = hidden_states.to(dtype)
    attention_mask = attention_mask.to(device)

    return hidden_states, attention_mask


def get_models_info() -> dict:
    """
    Get information about SBXL model components
    
    Returns:
        Dictionary with model information
    """
    return {
        "text_encoder": SAKIKO_TEXT_ENCODER_ID,
        "text_encoder_max_length": 384,
        "text_encoder_hidden_size": 1024,
        "unet_params": sbxl_models.SBXLParams.get_base_config().__dict__,
        "vae": "Flux VAE",
        "vae_scale_factor": 0.3611,
        "vae_shift_factor": 0.1159,
    }


# ============================================================================
# Flow Sampling (Euler sampling for Rectified Flow)
# ============================================================================

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """
    Get timestep schedule for flow matching
    
    Args:
        num_steps: Number of sampling steps
        image_seq_len: Sequence length of image (height * width)
        base_shift: Base shift value
        max_shift: Maximum shift value
        shift: Whether to apply shift
    
    Returns:
        List of timesteps
    """
    # Extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)
    
    # Shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # Estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    
    return timesteps.tolist()


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    """Get linear function for schedule shifting"""
    import math
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    """Apply time shift to timesteps"""
    import math
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def do_sample(
    model: sbxl_models.SBXLUNet2DConditionModel,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timesteps: list[float],
    guidance_scale: float = 3.5,
    attention_mask: Optional[torch.Tensor] = None,
    neg_encoder_hidden_states: Optional[torch.Tensor] = None,
    neg_attention_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Perform flow-based sampling (Euler method)
    
    Args:
        model: SBXL UNet model
        img: Initial noise tensor
        img_ids: Image position IDs
        encoder_hidden_states: Text encoder outputs
        timesteps: Timestep schedule
        guidance_scale: Guidance scale for flow matching
        attention_mask: Attention mask for text encoder outputs
        neg_encoder_hidden_states: Negative prompt encoder outputs
        neg_attention_mask: Negative attention mask
        cfg_scale: Classifier-free guidance scale
    
    Returns:
        Denoised latents
    """
    # Prepare classifier-free guidance
    logger.info(f"guidance_scale: {guidance_scale}, cfg_scale: {cfg_scale}")
    do_cfg = neg_encoder_hidden_states is not None and (cfg_scale is not None and cfg_scale != 1.0)
    
    if do_cfg:
        logger.info("Using classifier-free guidance")
        b_encoder_hidden_states = torch.cat([neg_encoder_hidden_states, encoder_hidden_states], dim=0)
        if attention_mask is not None and neg_attention_mask is not None:
            b_attention_mask = torch.cat([neg_attention_mask, attention_mask], dim=0)
        else:
            b_attention_mask = None
    else:
        b_encoder_hidden_states = encoder_hidden_states
        b_attention_mask = attention_mask
    
    # Sampling loop
    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]
        
        t_vec = torch.full((img.shape[0],), t_curr * 1000, dtype=img.dtype, device=img.device)
        
        if do_cfg:
            # Predict with and without guidance
            img_double = torch.cat([img, img], dim=0)
            t_vec_double = torch.cat([t_vec, t_vec], dim=0)
            
            pred = model(
                sample=img_double,
                timestep=t_vec_double,
                encoder_hidden_states=b_encoder_hidden_states,
                attention_mask=b_attention_mask,
            )["sample"]
            
            # Split predictions
            pred_neg, pred_pos = pred.chunk(2, dim=0)
            
            # Apply CFG
            pred = pred_neg + cfg_scale * (pred_pos - pred_neg)
        else:
            pred = model(
                sample=img,
                timestep=t_vec,
                encoder_hidden_states=b_encoder_hidden_states,
                attention_mask=b_attention_mask,
            )["sample"]
        
        # Euler step
        img = img + (t_prev - t_curr) * pred
    
    return img


def prepare_latents(
    vae: flux_models.AutoEncoder,
    image: Optional[torch.Tensor],
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Prepare initial latents for sampling
    
    Args:
        vae: VAE model
        image: Optional input image for img2img
        batch_size: Batch size
        height: Height in pixels
        width: Width in pixels
        dtype: Data type
        device: Device
        seed: Random seed
    
    Returns:
        Initial latents
    """
    # Calculate latent dimensions
    latent_height = height // 8
    latent_width = width // 8
    
    if image is not None:
        # Encode image to latents
        with torch.no_grad():
            latents = vae.encode(image)
    else:
        # Generate random noise
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        latents = torch.randn(
            batch_size,
            16,  # Flux VAE channels
            latent_height,
            latent_width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    
    return latents


def decode_latents(
    vae: flux_models.AutoEncoder,
    latents: torch.Tensor,
) -> torch.Tensor:
    """
    Decode latents to images using Flux VAE
    
    Args:
        vae: VAE model
        latents: Latents to decode [batch, 16, height, width]
        vae_scale_factor: Scale factor (default: 0.3611 for Flux)
        vae_shift_factor: Shift factor (default: 0.1159 for Flux)
    
    Returns:
        Decoded images [batch, 3, height*8, width*8]
    """
    with torch.no_grad():
        images = vae.decode(latents)

    return images

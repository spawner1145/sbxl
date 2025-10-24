# SBXL Training Utilities
# Training helper functions for SBXL

import argparse
import math
import os
from typing import Optional, Tuple, List, Callable, Dict, Any

import numpy as np

from accelerate import Accelerator

import torch
import torch.nn.functional as F

from library import train_util, strategy_base, strategy_sbxl, sbxl_utils
from library.flux_train_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Sample image utilities
# ============================================================================


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: Optional[int],
    global_step: int,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoders: Optional[List[torch.nn.Module]],
    sample_prompts_te_outputs: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    controlnet=None,
    prompt_replacement: Optional[Tuple[str, str]] = None,
):
    """Generate sample images during training using cached or live text encodings."""

    if accelerator is None:
        raise ValueError("accelerator is required for sampling")
    if vae is None:
        raise ValueError("vae is required for sampling")

    def _get_module_device(module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except (StopIteration, AttributeError):
            return accelerator.device

    if global_step == 0:
        if not getattr(args, "sample_at_first", False):
            return
    else:
        every_n_steps = getattr(args, "sample_every_n_steps", None)
        every_n_epochs = getattr(args, "sample_every_n_epochs", None)
        if every_n_steps is None and every_n_epochs is None:
            return
        if every_n_epochs is not None:
            if epoch is None or epoch % every_n_epochs != 0:
                return
        elif every_n_steps is not None and global_step % every_n_steps != 0:
            return

    if not getattr(args, "sample_prompts", None):
        logger.warning("Sampling skipped: --sample_prompts not provided")
        return

    controlnet_original_device = None
    controlnet_was_training = False
    if controlnet is not None:
        controlnet_original_device = _get_module_device(controlnet)
        controlnet_was_training = controlnet.training
        controlnet.to(accelerator.device)
        controlnet.eval()

    if not accelerator.is_main_process:
        return

    logger.info("")
    logger.info(f"Generating SBXL sample images at step {global_step}")

    prompts = train_util.load_prompts(args.sample_prompts)
    if len(prompts) == 0:
        logger.warning("No prompts found for sampling")
        return

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if torch.cuda.is_available():
        try:
            cuda_rng_state = torch.cuda.get_rng_state()
        except RuntimeError:
            cuda_rng_state = None

    sample_out_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(sample_out_dir, exist_ok=True)

    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    if not isinstance(tokenize_strategy, strategy_sbxl.SBXLTokenizeStrategy):
        raise RuntimeError("SBXL tokenize strategy not initialised")
    if not isinstance(text_encoding_strategy, strategy_sbxl.SBXLTextEncodingStrategy):
        raise RuntimeError("SBXL text encoding strategy not initialised")

    text_encoder = text_encoders[0] if text_encoders else None
    text_encoder_original_device = None
    text_encoder_was_training = False
    if text_encoder is not None:
        text_encoder_original_device = _get_module_device(text_encoder)
        text_encoder_was_training = text_encoder.training
        text_encoder.to(accelerator.device)
        text_encoder.eval()

    vae_original_device = _get_module_device(vae)
    vae_was_training = getattr(vae, "training", False)
    vae.to(accelerator.device)
    if hasattr(vae, "eval"):
        vae.eval()

    was_training = unet.training
    unet.eval()

    batch_size = getattr(args, "sample_batch_size", None) or getattr(args, "train_batch_size", 1) or 1
    total_saved = 0

    def get_text_encoder_outputs(prompt_text: str, is_negative: bool):
        cached = None
        if sample_prompts_te_outputs is not None:
            cached = sample_prompts_te_outputs.get(prompt_text)
        if cached is not None:
            hidden_states, input_ids, attn_mask = cached
            return (
                hidden_states.to(dtype=unet.dtype, device=accelerator.device),
                input_ids.to(accelerator.device),
                attn_mask.to(accelerator.device) if attn_mask is not None else None,
            )

        if text_encoder is None:
            raise RuntimeError(
                "Text encoder outputs not cached and text encoder unavailable; cannot encode prompts"
            )

        tokens_and_masks = tokenize_strategy.tokenize(prompt_text, is_negative=is_negative)
        hidden_states, _, attention_mask = text_encoding_strategy.encode_tokens(
            tokenize_strategy,
            [text_encoder],
            tokens_and_masks,
        )

        return (
            hidden_states.to(dtype=unet.dtype, device=accelerator.device),
            tokens_and_masks[0].to(accelerator.device),
            attention_mask.to(accelerator.device) if attention_mask is not None else None,
        )

    from PIL import Image

    def load_controlnet_tensor(path: str, width: int, height: int) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(device=accelerator.device, dtype=unet.dtype)

    try:
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]

            for batch_index, prompt_dict in enumerate(prompt_batch):
                prompt = prompt_dict.get("prompt", "")
                negative_prompt = prompt_dict.get("negative_prompt", "")
                if prompt_replacement is not None:
                    prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
                    if negative_prompt:
                        negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])
                sample_steps = int(prompt_dict.get("sample_steps", getattr(args, "sample_steps", 28)))
                cfg_scale = float(prompt_dict.get("scale", getattr(args, "sample_guidance_scale", 3.5)))
                height = int(prompt_dict.get("height", getattr(args, "sample_height", 1024)))
                width = int(prompt_dict.get("width", getattr(args, "sample_width", 1024)))
                seed = prompt_dict.get("seed")
                conditioning_scale = float(
                    prompt_dict.get("conditioning_scale", getattr(args, "conditioning_scale", 1.0))
                )
                if seed is None and getattr(args, "seed", None) is not None:
                    seed = args.seed + total_saved

                if not prompt:
                    logger.warning("Skipping empty prompt in sampling batch")
                    continue

                controlnet_cond = None
                controlnet_image_path = prompt_dict.get("controlnet_image")
                if controlnet is not None:
                    if controlnet_image_path is None:
                        logger.warning("ControlNet provided but prompt missing 'controlnet_image'; skipping conditioning")
                    else:
                        try:
                            controlnet_cond = load_controlnet_tensor(controlnet_image_path, width, height)
                        except Exception as ex:
                            logger.warning(f"Failed to load ControlNet image '{controlnet_image_path}': {ex}")
                            controlnet_cond = None
                    if controlnet_cond is None:
                        logger.warning("ControlNet conditioning unavailable; proceeding without ControlNet for this prompt")

                txt_emb, _, txt_mask = get_text_encoder_outputs(prompt, is_negative=False)
                neg_txt_emb = None
                neg_txt_mask = None
                if cfg_scale != 1.0 and negative_prompt:
                    neg_txt_emb, _, neg_txt_mask = get_text_encoder_outputs(negative_prompt, is_negative=True)

                latent_height, latent_width = calculate_latent_sizes(height, width, vae_scale_factor=8)
                noise = sbxl_utils.get_noise(
                    batch_size=1,
                    height=latent_height,
                    width=latent_width,
                    device=accelerator.device,
                    dtype=unet.dtype,
                    seed=seed,
                )

                timesteps = get_schedule(sample_steps, latent_height * latent_width, shift=True)

                with torch.no_grad():
                    latents = euler_sample(
                        model=unet,
                        img=noise,
                        txt_emb=txt_emb,
                        txt_mask=txt_mask,
                        timesteps=timesteps,
                        guidance_scale=cfg_scale,
                        neg_txt_emb=neg_txt_emb,
                        neg_txt_mask=neg_txt_mask,
                        controlnet=controlnet if controlnet_cond is not None else None,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=conditioning_scale,
                    )

                    latents = latents.to(device=accelerator.device, dtype=vae.dtype)
                    images = decode_latents(
                        vae=vae,
                        latents=latents,
                    )

                images = images.clamp(-1, 1)
                images = (images + 1) / 2
                images = images.permute(0, 2, 3, 1).cpu().float().numpy()
                images = (images * 255).round().astype("uint8")

                pil_image = Image.fromarray(images[0])

                filename = (
                    f"sbxl_e{epoch:04d}_s{global_step:06d}_b{start + batch_index:02d}.png"
                    if epoch is not None
                    else f"sbxl_s{global_step:06d}_b{start + batch_index:02d}.png"
                )
                output_path = os.path.join(sample_out_dir, filename)
                pil_image.save(output_path)
                logger.info(f"Saved sample image: {output_path}")
                total_saved += 1

    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(cuda_rng_state)
            except RuntimeError:
                pass

        if was_training:
            unet.train()
        if text_encoder is not None:
            if text_encoder_was_training:
                text_encoder.train()
            else:
                text_encoder.eval()
            if text_encoder_original_device is not None:
                text_encoder.to(text_encoder_original_device)
        if hasattr(vae, "train"):
            vae.train(vae_was_training)
        vae.to(vae_original_device)
        if controlnet is not None:
            if controlnet_was_training:
                controlnet.train()
            else:
                controlnet.eval()
            if controlnet_original_device is not None:
                controlnet.to(controlnet_original_device)

    logger.info(f"SBXL sampling complete, saved {total_saved} images")


# ============================================================================
# Flow Matching / Sampling utilities (adapted from Flux)
# ============================================================================

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    """Time shift function for flow sampling"""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    """Get linear function for schedule"""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    """
    Get timestep schedule for flow sampling - copied from Flux
    
    Args:
        num_steps: Number of sampling steps
        image_seq_len: Image sequence length
        base_shift: Base shift value
        max_shift: Maximum shift value
        shift: Whether to apply shift
    
    Returns:
        List of timesteps
    """
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise_step(
    model: torch.nn.Module,
    img: torch.Tensor,
    txt_emb: torch.Tensor,
    txt_mask: Optional[torch.Tensor],
    timestep: float,
    guidance_scale: float = 1.0,
    neg_txt_emb: Optional[torch.Tensor] = None,
    neg_txt_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[torch.nn.Module] = None,
    controlnet_cond: Optional[torch.Tensor] = None,
    conditioning_scale: float = 1.0,
) -> torch.Tensor:
    """
    Single denoising step for flow sampling
    
    Args:
        model: SBXL UNet model
        img: Current image latents
        txt_emb: Text embeddings
        txt_mask: Text attention mask
        timestep: Current timestep
        guidance_scale: Guidance scale for CFG
        neg_txt_emb: Negative text embeddings for CFG
        neg_txt_mask: Negative text attention mask
    
    Returns:
        Predicted velocity
    """
    # Prepare timestep (no manual scaling, let embedding handle it)
    t = torch.tensor([timestep], device=img.device, dtype=img.dtype)
    t = t.expand(img.shape[0])
    
    # Classifier-free guidance
    if controlnet is not None and controlnet_cond is None:
        raise ValueError("ControlNet provided without conditioning input")

    if neg_txt_emb is not None and guidance_scale != 1.0:
        # Concatenate for CFG
        img_input = torch.cat([img, img], dim=0)
        txt_input = torch.cat([neg_txt_emb, txt_emb], dim=0)
        t_input = torch.cat([t, t], dim=0)
        
        if txt_mask is not None and neg_txt_mask is not None:
            mask_input = torch.cat([neg_txt_mask, txt_mask], dim=0)
        else:
            mask_input = None

        if controlnet is not None:
            cond_input = controlnet_cond
            if cond_input.shape[0] == 1:
                cond_input = cond_input.repeat(2, 1, 1, 1)
            elif cond_input.shape[0] == img.shape[0]:
                cond_input = torch.cat([cond_input, cond_input], dim=0)
            down_residuals, mid_residual = controlnet(
                sample=img_input,
                timestep=t_input,
                encoder_hidden_states=txt_input,
                attention_mask=mask_input,
                controlnet_cond=cond_input,
                conditioning_scale=conditioning_scale,
            )
            down_residuals = list(down_residuals)
        else:
            down_residuals = None
            mid_residual = None
        
        # Predict
        pred = model(
            sample=img_input,
            timestep=t_input,
            encoder_hidden_states=txt_input,
            attention_mask=mask_input,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]
        
        # Split and apply guidance
        pred_neg, pred_pos = pred.chunk(2, dim=0)
        pred = pred_neg + guidance_scale * (pred_pos - pred_neg)
    else:
        if controlnet is not None:
            down_residuals, mid_residual = controlnet(
                sample=img,
                timestep=t,
                encoder_hidden_states=txt_emb,
                attention_mask=txt_mask,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
            )
            down_residuals = list(down_residuals)
        else:
            down_residuals = None
            mid_residual = None
        # No CFG
        pred = model(
            sample=img,
            timestep=t,
            encoder_hidden_states=txt_emb,
            attention_mask=txt_mask,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]
    
    return pred


def euler_sample(
    model: torch.nn.Module,
    img: torch.Tensor,
    txt_emb: torch.Tensor,
    txt_mask: Optional[torch.Tensor],
    timesteps: List[float],
    guidance_scale: float = 1.0,
    neg_txt_emb: Optional[torch.Tensor] = None,
    neg_txt_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[torch.nn.Module] = None,
    controlnet_cond: Optional[torch.Tensor] = None,
    conditioning_scale: float = 1.0,
    progress_bar: Optional[Any] = None,
) -> torch.Tensor:
    """
    Euler sampling for flow matching - adapted from Flux denoise function
    
    Args:
        model: SBXL UNet model
        img: Initial noise
        txt_emb: Text embeddings
        txt_mask: Text attention mask
        timesteps: Timestep schedule
        guidance_scale: Guidance scale for CFG
        neg_txt_emb: Negative text embeddings
        neg_txt_mask: Negative text attention mask
    
    Returns:
        Generated image latents
    """
    model.eval()
    
    # prepare classifier free guidance
    do_cfg = neg_txt_emb is not None and guidance_scale != 1.0
    
    iterator = range(len(timesteps) - 1)
    if progress_bar is not None:
        iterator = progress_bar(iterator, desc="Sampling", total=len(timesteps) - 1)
    
    for i in iterator:
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        
        # Predict velocity using denoise_step
        with torch.no_grad():
            pred = denoise_step(
                model,
                img,
                txt_emb,
                txt_mask,
                t_curr,
                guidance_scale,
                neg_txt_emb,
                neg_txt_mask,
                controlnet=controlnet,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
            )
        
        # Euler step (copied from Flux)
        img = img + (t_next - t_curr) * pred
    
    return img


# ============================================================================
# Training utilities
# ============================================================================

def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    weighting_scheme: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
) -> torch.Tensor:
    """
    Sample timesteps for training based on weighting scheme
    
    Args:
        batch_size: Batch size
        device: Device to place tensors on
        dtype: Data type
        weighting_scheme: Weighting scheme ("uniform", "logit_normal", "mode")
        logit_mean: Mean for logit normal distribution
        logit_std: Standard deviation for logit normal distribution
        mode_scale: Scale for mode distribution
    
    Returns:
        Sampled timesteps in [0, 1] range
    """
    if weighting_scheme == "logit_normal":
        # Sample from logit normal distribution
        u = torch.randn((batch_size,), device=device, dtype=dtype)
        u = u * logit_std + logit_mean
        sigmas = torch.sigmoid(u)
    elif weighting_scheme == "mode":
        # Sample from mode distribution (heavier tails)
        u = torch.randn((batch_size,), device=device, dtype=dtype)
        u = u * logit_std + logit_mean
        # Apply mode scaling
        u = u * mode_scale
        sigmas = torch.sigmoid(u)
    else:
        # Uniform sampling
        sigmas = torch.rand((batch_size,), device=device, dtype=dtype)
    
    return sigmas


def compute_loss_weighting(
    timesteps: torch.Tensor,
    weighting_scheme: str = "uniform",
) -> torch.Tensor:
    """
    Compute loss weighting based on timesteps
    
    Args:
        timesteps: Timesteps
        weighting_scheme: Weighting scheme
    
    Returns:
        Loss weights
    """
    if weighting_scheme == "uniform":
        return torch.ones_like(timesteps)
    elif weighting_scheme == "logit_normal":
        # Reciprocal of probability density for importance sampling
        return 1.0 / (timesteps * (1 - timesteps) + 1e-5)
    else:
        return torch.ones_like(timesteps)


def prepare_conditioning(
    txt_emb: torch.Tensor,
    txt_mask: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare conditioning for training
    
    Args:
        txt_emb: Text embeddings
        txt_mask: Text attention mask
        batch_size: Batch size
    
    Returns:
        Prepared text embeddings and mask
    """
    # batch对齐，支持不能整除的情况
    if txt_emb.shape[0] != batch_size:
        repeat_times = math.ceil(batch_size / txt_emb.shape[0])
        txt_emb = txt_emb.repeat(repeat_times, 1, 1)[:batch_size]
    if txt_mask is not None and txt_mask.shape[0] != batch_size:
        repeat_times = math.ceil(batch_size / txt_mask.shape[0])
        txt_mask = txt_mask.repeat(repeat_times, 1)[:batch_size]
    return txt_emb, txt_mask


def calculate_latent_sizes(
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> Tuple[int, int]:
    """
    Calculate latent sizes from image dimensions
    
    Args:
        height: Image height
        width: Image width
        vae_scale_factor: VAE downsampling factor
    
    Returns:
        Latent height and width
    """
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    return latent_height, latent_width


def prepare_latents(
    vae,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Encode images to latents using VAE
    
    Args:
        vae: VAE model
        images: Images tensor
        vae_scale_factor: VAE scale factor
        vae_shift_factor: VAE shift factor
    
    Returns:
        Encoded latents
    """
    with torch.no_grad():
        enc = vae.encode(images)
        if hasattr(enc, "latent_dist"):
            latents = enc.latent_dist.sample()
            scaling = getattr(vae, "scaling_factor", 1.0)
            latents = latents * scaling
        else:
            latents = enc
    return latents


def decode_latents(
    vae,
    latents: torch.Tensor,
) -> torch.Tensor:
    """
    Decode latents to images using VAE
    
    Args:
        vae: VAE model
        latents: Latents tensor
    
    Returns:
        Decoded images
    """
    with torch.no_grad():
        dec = vae.decode(latents.to(vae.dtype))
        if hasattr(dec, "sample"):
            images = dec.sample
        else:
            images = dec
    return images


def get_noisy_latents(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Add noise to latents for training (flow matching)
    
    Args:
        clean_latents: Clean latents
        noise: Noise tensor
        timesteps: Timesteps for each sample
    
    Returns:
        Noisy latents
    """
    t = timesteps.view(-1, 1, 1, 1)
    noisy_latents = (1 - t) * clean_latents + t * noise
    return noisy_latents


def get_target_for_loss(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Get target for loss computation (velocity for flow matching)
    
    Args:
        clean_latents: Clean latents
        noise: Noise tensor
        timesteps: Timesteps
    
    Returns:
        Target velocities
    """
    target = noise - clean_latents
    return target


def apply_snr_weight(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    min_snr_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply SNR weighting to loss values.

    Args:
        loss: Loss tensor
        timesteps: Timesteps used for weighting
        min_snr_gamma: Minimum SNR gamma value

    Returns:
        Weighted loss tensor
    """
    if min_snr_gamma is None:
        return loss
    
    # Calculate SNR
    snr = timesteps / (1 - timesteps + 1e-5)
    
    # Apply min-SNR weighting
    snr_weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=snr.device))
    snr_weight = snr_weight / snr
    
    return loss * snr_weight


def apply_model(
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    txt_emb: torch.Tensor,
    txt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply model to predict velocity
    
    Args:
        model: SBXL UNet model
        noisy_latents: Noisy latents
        timesteps: Timesteps
        txt_emb: Text embeddings
        txt_mask: Text attention mask
    
    Returns:
        Predicted velocity
    """
    pred = model(
        sample=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=txt_emb,
        attention_mask=txt_mask,
        return_dict=False,
    )[0]
    
    return pred


def compute_loss(
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    txt_emb: torch.Tensor,
    txt_mask: Optional[torch.Tensor],
    target: torch.Tensor,
    weighting_scheme: str = "uniform",
    min_snr_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute training loss
    
    Args:
        model: SBXL UNet model
        noisy_latents: Noisy latents
        timesteps: Timesteps
        txt_emb: Text embeddings
        txt_mask: Text attention mask
        target: Target velocities
        weighting_scheme: Weighting scheme
        min_snr_gamma: Min SNR gamma for weighting
    
    Returns:
        Loss value
    """
    # Predict
    pred = apply_model(model, noisy_latents, timesteps, txt_emb, txt_mask)
    
    # MSE loss
    loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    
    # Apply timestep weighting
    weights = compute_loss_weighting(timesteps, weighting_scheme)
    loss = loss * weights
    
    # Apply SNR weighting
    loss = apply_snr_weight(loss, timesteps, min_snr_gamma)
    
    return loss.mean()


def get_optimizer_params(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> list:
    """
    Get optimizer parameters with weight decay
    
    Args:
        model: Model
        learning_rate: Learning rate
        weight_decay: Weight decay
    
    Returns:
        List of parameter groups
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and normalization layers
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": learning_rate, "weight_decay": 0.0},
    ]


def save_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    metadata: Optional[dict] = None,
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        metadata: Optional metadata to save
    """
    from safetensors.torch import save_file
    
    state_dict = model.state_dict()
    
    # Convert metadata to string format for safetensors
    if metadata:
        metadata_str = {k: str(v) for k, v in metadata.items()}
    else:
        metadata_str = {}
    
    save_file(state_dict, save_path, metadata=metadata_str)
    logger.info(f"Saved checkpoint to {save_path}")


def get_noisy_model_input_and_timesteps(
    latents: torch.Tensor,
    noise: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get noisy model input and timesteps for flow matching training
    
    Args:
        latents: Clean latents
        noise: Noise to add
        args: Training arguments
        device: Device
        dtype: Data type

    Returns:
        Tuple of (noisy_model_input, timesteps, sigmas)
    """
    bsz = latents.shape[0]
    if bsz == 0:
        raise ValueError("Batch size must be greater than zero when sampling timesteps")

    num_timesteps = 1000  # Fixed for SBXL

    # Sample sigmas based on sampling strategy / weighting
    timestep_sampling = getattr(args, "timestep_sampling", "uniform")
    weighting_scheme = getattr(args, "weighting_scheme", None)
    logit_mean = getattr(args, "logit_mean", 0.0)
    logit_std = getattr(args, "logit_std", 1.0)
    mode_scale = getattr(args, "mode_scale", 1.29)

    if timestep_sampling == "uniform" and weighting_scheme in {"logit_normal", "mode"}:
        # Use sample_timesteps for consistency with other flow matching models
        sigmas = sample_timesteps(
            batch_size=bsz,
            device=device,
            dtype=dtype,
            weighting_scheme=weighting_scheme,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
    elif timestep_sampling == "sigmoid":
        sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
        sigmas = torch.sigmoid(sigmoid_scale * torch.randn((bsz,), device=device, dtype=dtype))
    elif timestep_sampling == "shift":
        # For shift sampling, we need to handle it directly since it uses discrete_flow_shift
        shift = getattr(args, "discrete_flow_shift", 3.0)
        sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
        sigmas = torch.randn((bsz,), device=device, dtype=dtype)
        sigmas = torch.sigmoid(sigmas * sigmoid_scale)
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    else:
        # Default sampling (uniform in [0, 1])
        sigmas = torch.rand((bsz,), device=device, dtype=dtype)

    timesteps = sigmas * num_timesteps

    # Flow matching: x_t = (1 - t) * x_1 + t * x_0 where x_1 is clean and x_0 is noise
    sigmas_broadcast = sigmas.view(-1, 1, 1, 1)
    noisy_model_input = (1.0 - sigmas_broadcast) * latents + sigmas_broadcast * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def apply_model_prediction_type(
    args,
    model_pred: torch.Tensor,
    noisy_model_input: torch.Tensor,
    sigmas: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply model prediction type transformation
    
    Args:
        args: Training arguments
        model_pred: Model prediction
        noisy_model_input: Noisy model input
        sigmas: Sigmas tensor
    
    Returns:
        Tuple of (transformed_pred, weighting)
    """
    weighting = None
    model_prediction_type = getattr(args, "model_prediction_type", "raw")

    if model_prediction_type == "raw":
        return model_pred, weighting

    if model_prediction_type == "additive":
        model_pred = model_pred + noisy_model_input
        return model_pred, weighting

    if model_prediction_type == "sigma_scaled":
        if sigmas is None:
            raise ValueError("Sigma-scaled prediction requires sigmas tensor")

        sigmas_broadcast = sigmas.view(-1, 1, 1, 1)
        model_pred = model_pred * (-sigmas_broadcast) + noisy_model_input
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=getattr(args, "weighting_scheme", None),
            sigmas=sigmas_broadcast,
        )
        weighting = weighting.to(model_pred.dtype)

    return model_pred, weighting

import argparse
import datetime
import math
import os
import random
from typing import Optional, List
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

from library.device_utils import init_ipex, get_preferred_device
from library.utils import setup_logging, str_to_dtype
from library import sbxl_models, sbxl_utils, sbxl_train_util, strategy_sbxl
import networks.lora_sbxl as lora_sbxl

init_ipex()
setup_logging()

import logging
logger = logging.getLogger(__name__)


def tokenize_prompt_with_system_prompt(tokenizer, prompt, system_prompt, max_length=384):
    """Tokenize prompt with system prompt using the provided tokenizer"""
    # Apply system prompt
    if system_prompt:
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
        full_prompt = prompt
    
    encodings = tokenizer(
        full_prompt,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    
    return (encodings.input_ids, encodings.attention_mask)


def generate_image(
    model: sbxl_models.SBXLUNet2DConditionModel,
    text_encoder: AutoModel,
    tokenizer: AutoTokenizer,
    vae: sbxl_models.flux_models.AutoEncoder,
    prompt: str,
    negative_prompt: Optional[str],
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: int,
    guidance_scale: float,
    system_prompt: str,
    dtype: torch.dtype,
    te_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    args: argparse.Namespace,
):
    """
    Generate an image using SBXL
    
    Args:
        model: SBXL UNet model
        text_encoder: Sakiko text encoder
        tokenizer: Sakiko tokenizer
        vae: Flux VAE
        prompt: Text prompt
        negative_prompt: Negative prompt for CFG
        seed: Random seed
        image_width: Image width
        image_height: Image height
        steps: Number of sampling steps
        guidance_scale: Guidance scale for CFG
        system_prompt: System prompt for text encoder
        dtype: Model dtype
        te_dtype: Text encoder dtype
        vae_dtype: VAE dtype
        args: Command line arguments
    """
    # Setup device and dtype
    device = get_preferred_device()
    if args.device:
        device = torch.device(args.device)
    
    # dtype was already resolved to actual torch dtype when passed to this function
    logger.info(f"Device: {device}, dtype: {dtype}, te_dtype: {te_dtype}, vae_dtype: {vae_dtype}")
    
    # Move models to device
    model.to(device, dtype=dtype)
    model.eval()
    
    text_encoder.to(device, dtype=te_dtype)
    text_encoder.eval()
    
    vae.to(device, dtype=vae_dtype)
    vae.eval()
    
    # Encode prompts
    logger.info("Encoding prompts...")
    
    # Use the provided tokenizer directly instead of creating SBXLTokenizeStrategy
    # which tries to download from HuggingFace
    tokens_and_masks = tokenize_prompt_with_system_prompt(tokenizer, prompt, system_prompt, max_length=384)
    encoding_strategy = strategy_sbxl.SBXLTextEncodingStrategy()
    with torch.no_grad():
        txt_conds = encoding_strategy.encode_tokens(None, [text_encoder], tokens_and_masks)
    
    # Tokenize negative prompt
    neg_tokens_and_masks = tokenize_prompt_with_system_prompt(tokenizer, negative_prompt or "", "", max_length=384)
    with torch.no_grad():
        neg_txt_conds = encoding_strategy.encode_tokens(None, [text_encoder], neg_tokens_and_masks)
    
    # Unpack outputs
    txt_emb, _, txt_mask = txt_conds
    neg_txt_emb, _, neg_txt_mask = neg_txt_conds
    
    # Convert text embeddings to model dtype
    txt_emb = txt_emb.to(dtype)
    if neg_txt_emb is not None:
        neg_txt_emb = neg_txt_emb.to(dtype)
    
    # Calculate latent dimensions
    latent_height, latent_width = sbxl_train_util.calculate_latent_sizes(
        image_height, image_width, vae_scale_factor=8
    )
    
    # Generate initial noise
    logger.info(f"Generating image {image_width}x{image_height} ({latent_width}x{latent_height} latents)...")
    
    img = sbxl_utils.get_noise(
        batch_size=1,
        height=latent_height,
        width=latent_width,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    
    # Get timestep schedule
    image_seq_len = latent_height * latent_width
    if hasattr(args, 'discrete_flow_shift') and args.discrete_flow_shift != 3.185:
        # Use legacy inference schedule if explicitly set to different value
        logger.info(f"Using legacy inference schedule (discrete_flow_shift={args.discrete_flow_shift})")
        timesteps = sbxl_train_util.get_schedule(
            num_steps=steps,
            image_seq_len=image_seq_len,
            base_shift=0.5,
            max_shift=1.15,
            shift=True,
        )
    else:
        # Use training-compatible shift (default 3.185)
        logger.info(f"Using training-compatible discrete_flow_shift={args.discrete_flow_shift}")
        timesteps = get_schedule_from_shift(
            num_steps=steps,
            shift=args.discrete_flow_shift,
        )
    
    # Sampling loop
    logger.info(f"Sampling with {steps} steps, guidance_scale={guidance_scale}...")
    
    img = sbxl_train_util.euler_sample(
        model=model,
        img=img,
        txt_emb=txt_emb,
        txt_mask=txt_mask,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        neg_txt_emb=neg_txt_emb,
        neg_txt_mask=neg_txt_mask,
        progress_bar=tqdm,
    )
    
    # Decode latents
    logger.info("Decoding latents...")
    img = img.to(vae_dtype)
    images = sbxl_train_util.decode_latents(
        vae=vae,
        latents=img,
    )
    
    # Convert to PIL Image
    images = images.clamp(-1, 1)
    images = (images + 1) / 2
    images = images.permute(0, 2, 3, 1).cpu().float().numpy()
    images = (images * 255).round().astype("uint8")
    
    pil_images = [Image.fromarray(img) for img in images]
    
    return pil_images


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def get_schedule_from_shift(
    num_steps: int,
    shift: float,
) -> list[float]:
    """
    Get timestep schedule using the same logic as FlowMatchEulerDiscreteScheduler
    
    Args:
        num_steps: Number of sampling steps
        shift: Discrete flow shift value (same as training)
    
    Returns:
        List of timesteps (0-1 range for inference)
    """
    # Use the same logic as FlowMatchEulerDiscreteScheduler
    timesteps_np = np.linspace(1, num_steps, num_steps, dtype=np.float32)[::-1].copy()
    timesteps = torch.from_numpy(timesteps_np).to(dtype=torch.float32)
    sigmas = timesteps / num_steps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    
    # sigmas are already in 0-1 range, add final 0
    sigmas_list = sigmas.tolist()
    sigmas_list.append(0.0)
    
    return sigmas_list


def main():
    parser = argparse.ArgumentParser(description="SBXL minimal inference")
    
    # Model paths
    parser.add_argument("--unet", type=str, required=True, help="Path to SBXL UNet checkpoint")
    parser.add_argument("--text_encoder", type=str, default=None, help="Path to Sakiko text encoder (default: from HuggingFace)")
    parser.add_argument("--vae", type=str, required=True, help="Path to Flux VAE checkpoint")
    
    # LoRA support
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--system_prompt", type=str, default=None, 
                        help="System prompt for text encoder (default: standard image generation prompt)")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=28, help="Number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for CFG")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.185, 
                        help="Discrete flow shift for timestep scheduling (should match training shift)")
    
    # Device and dtype
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--dtype", type=str, default="bf16", help="Model dtype")
    parser.add_argument("--te_dtype", type=str, default=None, help="Text encoder dtype (default: same as dtype)")
    parser.add_argument("--vae_dtype", type=str, default=None, help="VAE dtype (default: same as dtype)")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (number of images to generate)")
    
    # Cache
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, help="Cache directory for tokenizer")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace models")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true", help="Disable memory mapping when loading safetensors")
    parser.add_argument("--disable_mmap", action="store_true", help="Disable memory mapping (deprecated, use disable_mmap_load_safetensors)")
    parser.add_argument("--sbxl_params_file", type=str, default=None, help="Path to JSON/TOML file with SBXL architecture parameters")
    parser.add_argument(
        "--sbxl_param",
        type=str,
        nargs="*",
        default=None,
        help="Override SBXL architecture parameters (key=value)",
    )
    
    # Model offloading
    parser.add_argument("--offload", action="store_true", help="Offload models to CPU to save VRAM")
    parser.add_argument("--cpu_offload_checkpointing", action="store_true", help="Enable CPU offload for gradient checkpointing to save VRAM")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for generating multiple images")
    
    args = parser.parse_args()
    
    # Handle deprecated disable_mmap option
    if args.disable_mmap:
        args.disable_mmap_load_safetensors = True
    
    # Resolve dtypes
    dtype = str_to_dtype(args.dtype)
    te_dtype = str_to_dtype(args.te_dtype, dtype)
    vae_dtype = str_to_dtype(args.vae_dtype, dtype)
    
    logger.info(f"Dtypes: model={dtype}, text_encoder={te_dtype}, vae={vae_dtype}")
    
    # Determine loading device
    loading_device = "cpu" if args.offload else get_preferred_device()
    if args.device:
        loading_device = torch.device(args.device) if not args.offload else "cpu"
    
    # Load models
    sbxl_preset_overrides, sbxl_cli_overrides = sbxl_utils.get_sbxl_params_from_args(
        getattr(args, "sbxl_params_file", None),
        getattr(args, "sbxl_param", None),
    )

    logger.info(f"Loading SBXL UNet from {args.unet}...")
    model = sbxl_utils.load_sbxl_unet(
        args.unet,
        dtype=dtype,
        device=loading_device,
        disable_mmap=args.disable_mmap_load_safetensors,
        preset_overrides=sbxl_preset_overrides,
        cli_overrides=sbxl_cli_overrides,
    )
    
    logger.info("Loading text encoder...")
    text_encoder = sbxl_utils.load_text_encoder(
        args.text_encoder,
        dtype=te_dtype,
        device=loading_device,
        disable_mmap=args.disable_mmap_load_safetensors,
        cache_dir=args.tokenizer_cache_dir or args.cache_dir,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = sbxl_utils.load_tokenizer(
        args.text_encoder,
        cache_dir=args.tokenizer_cache_dir or args.cache_dir,
    )
    
    logger.info(f"Loading VAE from {args.vae}...")
    vae = sbxl_utils.load_vae(
        args.vae,
        dtype=vae_dtype,
        device=loading_device,
        disable_mmap=args.disable_mmap_load_safetensors,
    )
    
    # Enable CPU offload checkpointing if requested
    if args.cpu_offload_checkpointing:
        logger.info("Enabling CPU offload checkpointing...")
        model.enable_gradient_checkpointing(cpu_offload=True)
    
    # Ensure model is in correct dtype
    model = model.to(dtype)
    
    # Load LoRA weights
    lora_models: List[lora_sbxl.SBXLLoRANetwork] = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0
        
        logger.info(f"Loading LoRA weights from {weights_file} with multiplier {multiplier}")
        weights_sd = load_file(weights_file)
        
        # Check if weights contain text encoder LoRA
        has_text_encoder_lora = any(key.startswith("text_encoder_loras.") for key in weights_sd.keys())
        text_encoder_for_lora = [text_encoder] if has_text_encoder_lora else None
        
        # Create LoRA network from weights
        lora_model, _ = lora_sbxl.create_network_from_weights(
            multiplier=multiplier,
            file=None,
            vae=vae,
            text_encoder=text_encoder_for_lora,
            unet=model,
            weights_sd=weights_sd,
            for_inference=True,
        )
        
        if args.merge_lora_weights:
            logger.info("Merging LoRA weights to model...")
            merge_param = next(model.parameters(), None)
            merge_device = merge_param.device if merge_param is not None else "cpu"
            merge_dtype = merge_param.dtype if merge_param is not None else None
            lora_model.merge_to(text_encoder_for_lora or [], model, weights_sd, dtype=merge_dtype, device=merge_device)
        else:
            logger.info("Applying LoRA to model...")
            lora_model.apply_to(text_encoder_for_lora or [], model, dtype=dtype)
            # Load state dict
            info = lora_model.load_state_dict(weights_sd, strict=False)
            logger.info(f"Loaded LoRA state dict: {info}")
            lora_model.set_multiplier(multiplier)
        
        # Convert LoRA weights to the same dtype as the model
        for lora in lora_model.unet_loras + lora_model.text_encoder_loras:
            lora.lora_down.to(dtype)
            lora.lora_up.to(dtype)
            if hasattr(lora, 'alpha') and lora.alpha is not None:
                lora.alpha.data = lora.alpha.data.to(dtype)
        
        # Ensure the entire model is in the correct dtype after LoRA application
        model.to(dtype)
        if text_encoder_for_lora:
            for te in text_encoder_for_lora:
                te.to(dtype)
        
        lora_model.eval()
        
        lora_models.append(lora_model)
    
    # Use default system prompt if not specified
    system_prompt = args.system_prompt
    if system_prompt is None:
        system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
    
    if not args.interactive:
        # Single/batch generation mode
        all_images = []
        
        for i in range(args.batch_size):
            seed = args.seed + i if args.seed is not None else None
            logger.info(f"Generating image {i + 1}/{args.batch_size} with seed {seed}...")
            
            images = generate_image(
                model=model,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                seed=seed,
                image_width=args.width,
                image_height=args.height,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                system_prompt=system_prompt,
                dtype=dtype,
                te_dtype=te_dtype,
                vae_dtype=vae_dtype,
                args=args,
            )
            
            all_images.extend(images)
        
        # Save images
        if args.output is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir or "."
            os.makedirs(output_dir, exist_ok=True)

            for i, img in enumerate(all_images):
                output_path = os.path.join(output_dir, f"sbxl_{timestamp}_{i:04d}.png")
                img.save(output_path)
                logger.info(f"Saved: {output_path}")
        else:
            if args.batch_size == 1:
                output_path = args.output
                if not os.path.isabs(output_path):
                    output_path = os.path.join(args.output_dir or ".", output_path)
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                all_images[0].save(output_path)
                logger.info(f"Saved: {output_path}")
            else:
                output_path_base = args.output
                if not os.path.isabs(output_path_base):
                    output_path_base = os.path.join(args.output_dir or ".", output_path_base)
                base, ext = os.path.splitext(output_path_base)
                os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
                for i, img in enumerate(all_images):
                    output_path = f"{base}_{i:04d}{ext}"
                    img.save(output_path)
                    logger.info(f"Saved: {output_path}")
    else:
        # Interactive mode
        logger.info("Entering interactive mode.")
        
        width = args.width
        height = args.height
        steps = args.steps
        guidance_scale = args.guidance_scale
        current_system_prompt = system_prompt
        negative_prompt = args.negative_prompt
        
        print("\nInteractive Mode - Commands:")
        print("  Enter prompt to generate image")
        print("  Options: --w <width> --h <height> --s <steps> --d <seed> --g <guidance>")
        print("           --n <negative_prompt> (use '-' for empty)")
        print("           --sys <system_prompt> (use '-' for default)")
        print("           --m <lora_multipliers> (comma-separated for multiple LoRAs)")
        print("  Type 'exit' to quit\n")
        
        while True:
            user_input = input(">>> ")
            
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue
            
            # Parse options
            options = user_input.split("--")
            prompt = options[0].strip()
            
            if not prompt:
                print("Error: Empty prompt")
                continue
            
            # Reset defaults for each generation
            seed = None
            current_negative_prompt = negative_prompt
            
            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if not opt:
                        continue
                    
                    key, value = (opt.split(None, 1) + [""])[:2]
                    
                    if key == "w":
                        width = int(value)
                    elif key == "h":
                        height = int(value)
                    elif key == "s":
                        steps = int(value)
                    elif key == "d":
                        seed = int(value)
                    elif key == "g":
                        guidance_scale = float(value)
                    elif key == "n":
                        current_negative_prompt = value if value != "-" else ""
                    elif key == "sys":
                        current_system_prompt = value if value != "-" else system_prompt
                    elif key == "m":
                        multipliers = value.split(",")
                        if len(multipliers) != len(lora_models):
                            logger.error(f"Invalid number of multipliers, expected {len(lora_models)}")
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(multipliers[i].strip()))
                    else:
                        logger.warning(f"Unknown option: --{key}")
                
                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid value for option --{key}: '{value}'. Error: {e}")
                    continue
            
            # Generate image
            try:
                images = generate_image(
                    model=model,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    prompt=prompt,
                    negative_prompt=current_negative_prompt if current_negative_prompt else None,
                    seed=seed,
                    image_width=width,
                    image_height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    system_prompt=current_system_prompt,
                    dtype=dtype,
                    te_dtype=te_dtype,
                    vae_dtype=vae_dtype,
                    args=args,
                )
                
                # Save image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                seed_suffix = f"_{seed}" if seed is not None else ""
                interactive_dir = os.path.join(args.output_dir or ".", "interactive")
                os.makedirs(interactive_dir, exist_ok=True)
                output_path = os.path.join(interactive_dir, f"image_{timestamp}{seed_suffix}.png")
                images[0].save(output_path)
                logger.info(f"Saved: {output_path}")
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

import argparse
import datetime
import gc
from importlib.util import find_spec
import random
import os
import re
import time
import copy
from types import ModuleType, SimpleNamespace
from typing import Tuple, Optional, List, Any, Dict, Union

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from library import hunyuan_image_models, hunyuan_image_text_encoder, hunyuan_image_utils
from library import hunyuan_image_vae
from library.hunyuan_image_vae import HunyuanVAE2D
from library.device_utils import clean_memory_on_device, synchronize_device
from library.safetensors_utils import mem_eff_save_file
from networks import lora_hunyuan_image


lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder 1 (Qwen2.5-VL) directory or path")
    parser.add_argument("--byt5", type=str, default=None, help="ByT5 Text Encoder 2 directory or path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier free guidance. Default is 3.5."
    )
    parser.add_argument(
        "--apg_start_step_ocr",
        type=int,
        default=38,
        help="Starting step for Adaptive Projected Guidance (APG) for image with text. Default is 38. Should be less than infer_steps, usually near the end.",
    )
    parser.add_argument(
        "--apg_start_step_general",
        type=int,
        default=5,
        help="Starting step for Adaptive Projected Guidance (APG) for general image. Default is 5. Should be less than infer_steps, usually near the beginning.",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Guidance rescale factor for steps without APG, 0.0 to 1.0. Default is 0.0 (no rescale).",
    )
    parser.add_argument(
        "--guidance_rescale_apg",
        type=float,
        default=0.0,
        help="Guidance rescale factor for steps with APG, 0.0 to 1.0. Default is 0.0 (no rescale).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt for generation, default is empty string")
    parser.add_argument("--image_size", type=int, nargs=2, default=[2048, 2048], help="image size, height and width")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Shift factor for flow matching schedulers. Default is 5.0.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")

    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoders")
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,  # default is None (no chunking)
        help="Chunk size for VAE decoding to reduce memory usage. Default is None (no chunking). 16 is recommended if enabled"
        " / メモリ使用量を減らすためのVAEデコードのチャンクサイズ。デフォルトはNone（チャンクなし）。有効にする場合は16程度を推奨。",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument(
        "--lycoris", action="store_true", help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}"
    )

    # arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        # elif option == "i":
        #     overrides["image_path"] = value
        # elif option == "im":
        #     overrides["image_mask_path"] = value
        # elif option == "cn":
        #     overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value
        # elif option == "ci":  # control_image_path
        #     overrides["control_image_path"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

    return height, width


# region Model


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> hunyuan_image_models.HYImageDiffusionTransformer:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        qwen_image_model.HYImageDiffusionTransformer: DiT model instance
    """
    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    loading_device = "cpu"
    if args.blocks_to_swap == 0 and not args.lycoris:
        loading_device = device

    # load LoRA weights
    if not args.lycoris and args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            # lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8

    model = hunyuan_image_models.load_hunyuan_image_model(
        device,
        args.dit,
        args.attn_mode,
        True,  # enable split_attn to trim masked tokens
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
    )

    # merge LoRA weights
    if args.lycoris:
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            merge_lora_weights(lora_hunyuan_image, model, args, device)

        if args.fp8_scaled:
            # load state dict as-is and optimize to fp8
            state_dict = model.state_dict()

            # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
            move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
            state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=False)  # args.fp8_fast)

            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded FP8 optimized weights: {info}")

    # if we only want to save the model, we can skip the rest
    if args.save_merged_model:
        return None

    if not args.fp8_scaled:
        # simple cast to dit_weight_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    # if args.compile:
    #     compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
    #     logger.info(
    #         f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
    #     )
    #     torch._dynamo.config.cache_size_limit = 32
    #     for i in range(len(model.blocks)):
    #         model.blocks[i] = torch.compile(
    #             model.blocks[i],
    #             backend=compile_backend,
    #             mode=compile_mode,
    #             dynamic=compile_dynamic.lower() in "true",
    #             fullgraph=compile_fullgraph.lower() in "true",
    #         )

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def merge_lora_weights(
    lora_module: ModuleType,
    model: torch.nn.Module,
    lora_weights: List[str],
    lora_multipliers: List[float],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    device: torch.device,
    lycoris: bool = False,
    save_merged_model: Optional[str] = None,
    converter: Optional[callable] = None,
) -> None:
    """merge LoRA weights to the model

    Args:
        lora_module: LoRA module, e.g. lora_wan
        model: DiT model
        lora_weights: paths to LoRA weights
        lora_multipliers: multipliers for LoRA weights
        include_patterns: regex patterns to include LoRA modules
        exclude_patterns: regex patterns to exclude LoRA modules
        device: torch.device
        lycoris: use LyCORIS
        save_merged_model: path to save merged model, if specified, no inference will be performed
        converter: Optional[callable] = None
    """
    if lora_weights is None or len(lora_weights) == 0:
        return

    for i, lora_weight in enumerate(lora_weights):
        if lora_multipliers is not None and len(lora_multipliers) > i:
            lora_multiplier = lora_multipliers[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)
        if converter is not None:
            weights_sd = converter(weights_sd)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if include_patterns is not None and len(include_patterns) > i:
            include_pattern = include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
        if exclude_patterns is not None and len(exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = exclude_patterns[i]
            regex_exclude = re.compile(exclude_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
            logger.info(
                f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
            )
        if len(weights_sd) != original_key_count:
            remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
            remaining_keys.sort()
            logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
            if len(weights_sd) == 0:
                logger.warning("No keys left after filtering.")

        if lycoris:
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=model,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
        else:
            network = lora_module.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # save model here before casting to dit_weight_dtype
    if save_merged_model:
        logger.info(f"Saving merged model to {save_merged_model}")
        mem_eff_save_file(model.state_dict(), save_merged_model)  # save_file needs a lot of memory
        logger.info("Merged model saved")


# endregion


def decode_latent(vae: HunyuanVAE2D, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")

    vae.to(device)
    with torch.no_grad():
        latent = latent / vae.scaling_factor  # scale latent back to original range
        pixels = vae.decode(latent.to(device, dtype=vae.dtype))
    pixels = pixels.to("cpu", dtype=torch.float32)  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for T2I: LLM encoding."""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    vl_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        tokenizer_vlm = shared_models.get("tokenizer_vlm")
        text_encoder_vlm = shared_models.get("text_encoder_vlm")
        tokenizer_byt5 = shared_models.get("tokenizer_byt5")
        text_encoder_byt5 = shared_models.get("text_encoder_byt5")

        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        vl_dtype = torch.bfloat16  # Default dtype for Text Encoder
        tokenizer_vlm, text_encoder_vlm = hunyuan_image_text_encoder.load_qwen2_5_vl(
            args.text_encoder, dtype=vl_dtype, device=vl_device, disable_mmap=True
        )
        tokenizer_byt5, text_encoder_byt5 = hunyuan_image_text_encoder.load_byt5(
            args.byt5, dtype=torch.float16, device=vl_device, disable_mmap=True
        )

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder_original_device = text_encoder_vlm.device if text_encoder_vlm else None

    # Ensure text_encoder is not None before proceeding
    if not text_encoder_vlm or not tokenizer_vlm or not tokenizer_byt5 or not text_encoder_byt5:
        raise ValueError("Text encoder or tokenizer is not loaded properly.")

    # Define a function to move models to device if needed
    # This is to avoid moving models if not needed, especially in interactive mode
    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving DiT and Text Encoder to appropriate device: {device} or CPU")
        if shared_models and "model" in shared_models:  # DiT model is shared
            if args.blocks_to_swap > 0:
                logger.info("Waiting for 5 seconds to finish block swap")
                time.sleep(5)
            model = shared_models["model"]
            model.to("cpu")
            clean_memory_on_device(device)  # clean memory on device before moving models

        text_encoder_vlm.to(vl_device)  # If text_encoder_cpu is True, this will be CPU
        text_encoder_byt5.to(vl_device)

    logger.info("Encoding prompt with Text Encoder")

    prompt = args.prompt
    cache_key = prompt
    if cache_key in conds_cache:
        embed, mask, embed_byt5, mask_byt5, ocr_mask = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        with torch.no_grad():
            embed, mask = hunyuan_image_text_encoder.get_qwen_prompt_embeds(tokenizer_vlm, text_encoder_vlm, prompt)
            ocr_mask, embed_byt5, mask_byt5 = hunyuan_image_text_encoder.get_glyph_prompt_embeds(
                tokenizer_byt5, text_encoder_byt5, prompt
            )
        embed = embed.cpu()
        mask = mask.cpu()
        embed_byt5 = embed_byt5.cpu()
        mask_byt5 = mask_byt5.cpu()

        conds_cache[cache_key] = (embed, mask, embed_byt5, mask_byt5, ocr_mask)

    negative_prompt = args.negative_prompt
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed, negative_mask, negative_embed_byt5, negative_mask_byt5, negative_ocr_mask = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        with torch.no_grad():
            negative_embed, negative_mask = hunyuan_image_text_encoder.get_qwen_prompt_embeds(
                tokenizer_vlm, text_encoder_vlm, negative_prompt
            )
            negative_ocr_mask, negative_embed_byt5, negative_mask_byt5 = hunyuan_image_text_encoder.get_glyph_prompt_embeds(
                tokenizer_byt5, text_encoder_byt5, negative_prompt
            )
        negative_embed = negative_embed.cpu()
        negative_mask = negative_mask.cpu()
        negative_embed_byt5 = negative_embed_byt5.cpu()
        negative_mask_byt5 = negative_mask_byt5.cpu()

        conds_cache[cache_key] = (negative_embed, negative_mask, negative_embed_byt5, negative_mask_byt5, negative_ocr_mask)

    if not (shared_models and "text_encoder_vlm" in shared_models):  # if loaded locally
        # There is a bug text_encoder is not freed from GPU memory when text encoder is fp8
        del tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5
        gc.collect()  # This may force Text Encoder to be freed from GPU memory
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder_vlm:
            text_encoder_vlm.to(text_encoder_original_device)
        if text_encoder_byt5:
            text_encoder_byt5.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "mask": mask, "embed_byt5": embed_byt5, "mask_byt5": mask_byt5, "ocr_mask": ocr_mask, "prompt": prompt}
    arg_null = {
        "embed": negative_embed,
        "mask": negative_mask,
        "embed_byt5": negative_embed_byt5,
        "mask_byt5": negative_mask_byt5,
        "ocr_mask": negative_ocr_mask,
        "prompt": negative_prompt,
    }

    return arg_c, arg_null


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_image_data: Optional dictionary with precomputed image data
        precomputed_text_data: Optional dictionary with precomputed text data

    Returns:
        tuple: (HunyuanVAE2D model (vae) or None, torch.Tensor generated latent)
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, shared_models)

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        model = load_dit_model(args, device, dit_weight_dtype)

        # if we only want to save the model, we can skip the rest
        if args.save_merged_model:
            return None

        if shared_models is not None:
            shared_models["model"] = model
    else:
        # use shared model
        logger.info("Using shared DiT model.")
        model: hunyuan_image_models.HYImageDiffusionTransformer = shared_models["model"]
        model.move_to_device_except_swap_blocks(device)  # Handles block swap correctly
        model.prepare_block_swap_before_forward()

    return generate_body(args, model, context, context_null, device, seed)


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    model: hunyuan_image_models.HYImageDiffusionTransformer,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"].to(device, dtype=torch.bfloat16)
    mask = context["mask"].to(device, dtype=torch.bfloat16)
    embed_byt5 = context["embed_byt5"].to(device, dtype=torch.bfloat16)
    mask_byt5 = context["mask_byt5"].to(device, dtype=torch.bfloat16)
    ocr_mask = context["ocr_mask"]  # list of bool

    if context_null is None:
        context_null = context  # dummy for unconditional

    negative_embed = context_null["embed"].to(device, dtype=torch.bfloat16)
    negative_mask = context_null["mask"].to(device, dtype=torch.bfloat16)
    negative_embed_byt5 = context_null["embed_byt5"].to(device, dtype=torch.bfloat16)
    negative_mask_byt5 = context_null["mask_byt5"].to(device, dtype=torch.bfloat16)
    # negative_ocr_mask = context_null["ocr_mask"]  # list of bool

    # Prepare latent variables
    num_channels_latents = model.in_channels
    shape = (1, num_channels_latents, height // hunyuan_image_vae.VAE_SCALE_FACTOR, width // hunyuan_image_vae.VAE_SCALE_FACTOR)
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    logger.info(
        f"Embed: {embed.shape}, embed byt5: {embed_byt5.shape}, negative_embed: {negative_embed.shape}, negative embed byt5: {negative_embed_byt5.shape}, latents: {latents.shape}"
    )

    # Prepare timesteps
    timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)

    # Prepare Guider
    cfg_guider_ocr = hunyuan_image_utils.AdaptiveProjectedGuidance(
        guidance_scale=10.0,
        eta=0.0,
        adaptive_projected_guidance_rescale=10.0,
        adaptive_projected_guidance_momentum=-0.5,
        guidance_rescale=args.guidance_rescale_apg,
    )
    cfg_guider_general = hunyuan_image_utils.AdaptiveProjectedGuidance(
        guidance_scale=10.0,
        eta=0.0,
        adaptive_projected_guidance_rescale=10.0,
        adaptive_projected_guidance_momentum=-0.5,
        guidance_rescale=args.guidance_rescale_apg,
    )

    # Denoising loop
    do_cfg = args.guidance_scale != 1.0
    # print(f"embed shape: {embed.shape}, mean: {embed.mean()}, std: {embed.std()}")
    # print(f"embed_byt5 shape: {embed_byt5.shape}, mean: {embed_byt5.mean()}, std: {embed_byt5.std()}")
    # print(f"negative_embed shape: {negative_embed.shape}, mean: {negative_embed.mean()}, std: {negative_embed.std()}")
    # print(f"negative_embed_byt5 shape: {negative_embed_byt5.shape}, mean: {negative_embed_byt5.mean()}, std: {negative_embed_byt5.std()}")
    # print(f"latents shape: {latents.shape}, mean: {latents.mean()}, std: {latents.std()}")
    # print(f"mask shape: {mask.shape}, sum: {mask.sum()}")
    # print(f"mask_byt5 shape: {mask_byt5.shape}, sum: {mask_byt5.sum()}")
    # print(f"negative_mask shape: {negative_mask.shape}, sum: {negative_mask.sum()}")
    # print(f"negative_mask_byt5 shape: {negative_mask_byt5.shape}, sum: {negative_mask_byt5.sum()}")

    autocast_enabled = args.fp8

    with tqdm(total=len(timesteps), desc="Denoising steps") as pbar:
        for i, t in enumerate(timesteps):
            t_expand = t.expand(latents.shape[0]).to(torch.int64)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                noise_pred = model(latents, t_expand, embed, mask, embed_byt5, mask_byt5)

            if do_cfg:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    uncond_noise_pred = model(
                        latents, t_expand, negative_embed, negative_mask, negative_embed_byt5, negative_mask_byt5
                    )
                noise_pred = hunyuan_image_utils.apply_classifier_free_guidance(
                    noise_pred,
                    uncond_noise_pred,
                    ocr_mask[0],
                    args.guidance_scale,
                    i,
                    apg_start_step_ocr=args.apg_start_step_ocr,
                    apg_start_step_general=args.apg_start_step_general,
                    cfg_guider_ocr=cfg_guider_ocr,
                    cfg_guider_general=cfg_guider_general,
                    guidance_rescale=args.guidance_rescale,
                )

            # ensure latents dtype is consistent
            latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)

            pbar.update()

    return latents


def get_time_flag():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            # "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"

    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C

    image = Image.fromarray(x)
    image.save(os.path.join(save_path, f"{image_name}.png"))

    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: HunyuanVAE2D,
    latent: torch.Tensor,
    device: torch.device,
    original_base_name: Optional[str] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_name: original base name (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= hunyuan_image_vae.VAE_SCALE_FACTOR
    width *= hunyuan_image_vae.VAE_SCALE_FACTOR
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    if latent.ndim == 2:  # S,C. For packed latents from other inference scripts
        latent = latent.unsqueeze(0)
        height, width = check_inputs(args)  # Get height/width from args
        latent = latent.view(
            1, vae.latent_channels, height // hunyuan_image_vae.VAE_SCALE_FACTOR, width // hunyuan_image_vae.VAE_SCALE_FACTOR
        )

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        # save images
        if original_base_name is None:
            original_name = ""
        else:
            original_name = f"_{original_base_name}"
        save_images(image, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models (text/image encoders)
    """
    shared_models = {}
    # Load text encoders to CPU
    vl_dtype = torch.bfloat16  # Default dtype for Text Encoder
    tokenizer_vlm, text_encoder_vlm = hunyuan_image_text_encoder.load_qwen2_5_vl(
        args.text_encoder, dtype=vl_dtype, device="cpu", disable_mmap=True
    )
    tokenizer_byt5, text_encoder_byt5 = hunyuan_image_text_encoder.load_byt5(
        args.byt5, dtype=torch.float16, device="cpu", disable_mmap=True
    )
    shared_models["tokenizer_vlm"] = tokenizer_vlm
    shared_models["text_encoder_vlm"] = text_encoder_vlm
    shared_models["tokenizer_byt5"] = tokenizer_byt5
    shared_models["text_encoder_byt5"] = text_encoder_byt5
    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = hunyuan_image_vae.load_vae(args.vae, device="cpu", disable_mmap=True, chunk_size=args.vae_chunk_size)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)  # Validate each prompt's height/width

    # 2. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    # Text Encoder loaded to CPU by load_text_encoder
    vl_dtype = torch.bfloat16  # Default dtype for Text Encoder
    tokenizer_vlm, text_encoder_vlm_batch = hunyuan_image_text_encoder.load_qwen2_5_vl(
        args.text_encoder, dtype=vl_dtype, device="cpu", disable_mmap=True
    )
    tokenizer_byt5, text_encoder_byt5_batch = hunyuan_image_text_encoder.load_byt5(
        args.byt5, dtype=torch.float16, device="cpu", disable_mmap=True
    )

    # Text Encoder to device for this phase
    vl_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_vlm_batch.to(vl_device)  # Moved into prepare_text_inputs logic
    text_encoder_byt5_batch.to(vl_device)

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "tokenizer_vlm": tokenizer_vlm,
        "text_encoder_vlm": text_encoder_vlm_batch,  # on GPU if not text_encoder_cpu
        "tokenizer_byt5": tokenizer_byt5,
        "text_encoder_byt5": text_encoder_byt5_batch,  # on GPU if not text_encoder_cpu
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")

        # prepare_text_inputs will move text_encoders to device temporarily
        context, context_null = prepare_text_inputs(prompt_args_item, device, temp_shared_models_txt)
        text_data = {"context": context, "context_null": context_null}
        all_precomputed_text_data.append(text_data)

    # Models should be removed from device after prepare_text_inputs
    del tokenizer_batch, text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    # 3. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    dit_model = load_dit_model(first_prompt_args, device, dit_weight_dtype)  # Load directly to target device if possible

    if first_prompt_args.save_merged_model:
        logger.info("Merged DiT model saved. Skipping generation.")

    shared_models_for_generate = {"model": dit_model}  # Pass DiT via shared_models

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_text_data = all_precomputed_text_data[i]
            height, width = check_inputs(prompt_args_item)  # Get height/width for each prompt

            logger.info(f"Generating latent for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                # generate is called with precomputed data, so it won't load Text Encoders.
                # It will use the DiT model from shared_models_for_generate.
                latent = generate(prompt_args_item, gen_settings, shared_models_for_generate, current_text_data)

                if latent is None:  # and prompt_args_item.save_merged_model:  # Should be caught earlier
                    continue

                # Save latent if needed (using data from precomputed_image_data for H/W)
                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    save_latent(latent, prompt_args_item, height, width)

                all_latents.append(latent)
            except Exception as e:
                logger.error(f"Error generating latent for prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                all_latents.append(None)  # Add placeholder for failed generations
                continue

    # Free DiT model
    logger.info("Releasing DiT model from memory...")
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    del shared_models_for_generate["model"]
    del dit_model
    clean_memory_on_device(device)
    synchronize_device(device)  # Ensure memory is freed before loading VAE for decoding

    # 4. Decode latents and save outputs (using vae_for_batch)
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images using batched VAE...")
        vae_for_batch.to(device)  # Move VAE to device for decoding

        for i, latent in enumerate(all_latents):
            if latent is None:  # Skip failed generations
                logger.warning(f"Skipping decoding for prompt {i+1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i+1}/{len(all_latents)} for prompt: {current_args.prompt}")

            # if args.output_type is "latent_images", we already saved latent above.
            # so we skip saving latent here.
            if current_args.output_type == "latent_images":
                current_args.output_type = "images"

            # save_output expects latent to be [BCTHW] or [CTHW]. generate returns [BCTHW] (batch size 1).
            # latent[0] is correct if generate returns it with batch dim.
            # The latent from generate is (1, C, T, H, W)
            save_output(current_args, vae_for_batch, latent[0], device)  # Pass vae_for_batch

        vae_for_batch.to("cpu")  # Move VAE back to CPU

    del vae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}  # Initialize empty cache for interactive mode

    vae = hunyuan_image_vae.load_vae(args.vae, device="cpu", disable_mmap=True, chunk_size=args.vae_chunk_size)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError  # Exit on Ctrl+D or Ctrl+Z

                # Parse prompt
                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Generate latent
                # For interactive, precomputed data is None. shared_models contains text encoders.
                latent = generate(prompt_args, gen_settings, shared_models)

                # # If not one_frame_inference, move DiT model to CPU after generation
                # if prompt_args.blocks_to_swap > 0:
                #     logger.info("Waiting for 5 seconds to finish block swap")
                #     time.sleep(5)
                # model = shared_models.get("model")
                # model.to("cpu")  # Move DiT model to CPU after generation

                # Save latent and video
                # returned_vae from generate will be used for decoding here.
                save_output(prompt_args, vae, latent, device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        # assert len(args.latent_path) == 1, "Only one latent path is supported for now"

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        # latent = torch.stack(latents_list, dim=0)  # [N, ...], must be same shape

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]

            vae = hunyuan_image_vae.load_vae(args.vae, device=device, disable_mmap=True, chunk_size=args.vae_chunk_size)
            vae.eval()
            save_output(args, vae, latent, device, original_base_names[i])

    elif args.from_file:
        # Batch mode from file

        # Read prompts from file
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        # Process prompts
        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        process_batch_prompts(prompts_data, args)

    elif args.interactive:
        # Interactive mode
        process_interactive(args)

    else:
        # Single prompt mode (original behavior)

        # Generate latent
        gen_settings = get_generation_settings(args)

        # For single mode, precomputed data is None, shared_models is None.
        # generate will load all necessary models (Text Encoders, DiT).
        latent = generate(args, gen_settings)
        # print(f"Generated latent shape: {latent.shape}")
        # if args.save_merged_model:
        #     return

        clean_memory_on_device(device)

        # Save latent and video
        vae = hunyuan_image_vae.load_vae(args.vae, device="cpu", disable_mmap=True, chunk_size=args.vae_chunk_size)
        vae.eval()
        save_output(args, vae, latent, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()

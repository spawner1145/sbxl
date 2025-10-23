# SBXL Full Model Training Script
# Based on flux_train.py and sdxl_train.py
# For full fine-tuning of SBXL model

import argparse
import copy
import math
import os
import time
from multiprocessing import Value
from typing import Dict, List, Optional, Tuple, Union

try:
    import toml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    toml = None
import torch
import torch.nn as nn
from tqdm import tqdm

from library import utils
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from accelerate.utils import set_seed

import library.train_util as train_util
import library.sai_model_spec as sai_model_spec
from library import (
    deepspeed_utils,
    huggingface_util,
    sbxl_train_util,
    sbxl_utils,
    strategy_base,
    strategy_sbxl,
)
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

from library.config_util import (
    BlueprintGenerator,
    ConfigSanitizer,
)
from library.custom_train_functions import add_custom_train_arguments, apply_masked_loss


def train(args):
    """
    Main training function for SBXL full model training
    
    This implements the full training loop based on flux_train.py structure.
    """
    import os
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # Backward compatibility
    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
        )
        args.cache_text_encoder_outputs = True

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning(
            "cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
        )
        args.gradient_checkpointing = True

    cache_latents = args.cache_latents
    sample_prompts_te_outputs: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    use_dreambooth_method = args.in_json is None

    if args.output_dir is None:
        raise ValueError("--output_dir must be specified for SBXL training")

    save_format = (args.save_model_as or "safetensors").lower()
    if save_format not in {"safetensors", "ckpt", "pt"}:
        raise ValueError("--save_model_as must be one of safetensors, ckpt, pt for SBXL training")
    ckpt_ext = ".safetensors" if save_format == "safetensors" else ".ckpt" if save_format == "ckpt" else ".pt"

    if args.seed is not None:
        set_seed(args.seed)

    # Prepare caching strategy
    if args.cache_latents:
        latents_caching_strategy = strategy_sbxl.SBXLLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Prepare dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    f"ignore following options because config file is found: {', '.join(ignored)}"
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_sbxl.SBXLTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk,
                    args.text_encoder_batch_size if hasattr(args, 'text_encoder_batch_size') else args.vae_batch_size,
                    args.skip_cache_check,
                    False
                )
            )
        strategy_base.TokenizeStrategy.set_strategy(
            strategy_sbxl.SBXLTokenizeStrategy(
                max_length=args.max_token_length if hasattr(args, 'max_token_length') else 384,
                tokenizer_path=args.text_encoder_path if hasattr(args, 'text_encoder_path') and args.text_encoder_path else None,
                tokenizer_cache_dir=args.tokenizer_cache_dir if hasattr(args, 'tokenizer_cache_dir') else None,
                system_prompt=args.system_prompt if hasattr(args, 'system_prompt') else None,
            )
        )

        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return

    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option."
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

    # Prepare accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # Prepare dtype
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # Load models
    logger.info("Loading SBXL Models")

    sbxl_preset_overrides, sbxl_cli_overrides = sbxl_utils.get_sbxl_params_from_args(
        getattr(args, "sbxl_params_file", None),
        getattr(args, "sbxl_param", None),
    )
    
    # Load VAE for caching latents
    ae = None
    vae_path = getattr(args, 'sbxl_vae', args.vae)
    if cache_latents:
        logger.info(f"Loading VAE from {vae_path}")
        ae = sbxl_utils.load_vae(vae_path, weight_dtype, "cpu", args.disable_mmap_load_safetensors)
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.requires_grad_(False)
        ae.eval()

        train_dataset_group.new_cache_latents(ae, accelerator)

        ae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()
    else:
        logger.info(f"Loading VAE from {vae_path} for on-the-fly encoding")
        ae = sbxl_utils.load_vae(vae_path, weight_dtype, "cpu", args.disable_mmap_load_safetensors)
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.requires_grad_(False)
        ae.eval()

    # Load text encoder
    logger.info("Loading Sakiko text encoder")
    text_encoder = sbxl_utils.load_text_encoder(
        args.text_encoder_path if hasattr(args, 'text_encoder_path') and args.text_encoder_path else None,
        weight_dtype,
        "cpu",
        args.disable_mmap_load_safetensors,
        cache_dir=args.tokenizer_cache_dir if hasattr(args, 'tokenizer_cache_dir') else None,
    )

    # Load UNet
    logger.info(f"Loading SBXL UNet from {args.pretrained_model_name_or_path}")
    unet = sbxl_utils.load_sbxl_unet(
        args.pretrained_model_name_or_path,
        weight_dtype,
        "cpu",
        args.disable_mmap_load_safetensors,
        preset_overrides=sbxl_preset_overrides,
        cli_overrides=sbxl_cli_overrides,
    )
    args.sbxl_resolved_params = unet.params.copy()
    attention_backend = sbxl_utils.select_attention_backend_from_args(args)
    sbxl_utils.configure_attention_backend(unet, attention_backend)

    logger.info("Model loading complete")
    
    # Fix BatchNorm weights for BF16 compatibility
    if weight_dtype == torch.bfloat16:
        logger.info("Converting BatchNorm weights to float32 for BF16 compatibility")
        for module in unet.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()
                module.running_mean.data = module.running_mean.data.float()
                module.running_var.data = module.running_var.data.float()
    
    # Setup tokenization and text encoding strategies
    logger.info("Setting up tokenization and encoding strategies")
    tokenize_strategy = strategy_sbxl.SBXLTokenizeStrategy(
        max_length=args.max_token_length if hasattr(args, 'max_token_length') else 384,
        tokenizer_path=args.text_encoder_path if hasattr(args, 'text_encoder_path') and args.text_encoder_path else None,
        tokenizer_cache_dir=args.tokenizer_cache_dir if hasattr(args, 'tokenizer_cache_dir') else None,
        system_prompt=args.system_prompt if hasattr(args, 'system_prompt') else None,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    
    text_encoding_strategy = strategy_sbxl.SBXLTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)
    
    # Create noise scheduler for SBXL (Euler discrete flow matching)
    from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, 
        shift=getattr(args, 'discrete_flow_shift', 3.0)
    )
    
    train_text_encoder = getattr(args, 'train_text_encoder', False)

    # Cache text encoder outputs if needed
    if args.cache_text_encoder_outputs:
        text_encoder_outputs_caching_strategy = strategy_sbxl.SBXLTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size if hasattr(args, 'text_encoder_batch_size') else args.vae_batch_size,
            args.skip_cache_check,
            train_text_encoder,
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        
        # Cache text encoder outputs
        logger.info("Caching text encoder outputs")
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.eval()
        
        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([text_encoder], accelerator)

        if args.sample_prompts is not None and not train_text_encoder:
            logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

            tokenize_strategy_cached = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy_cached = strategy_base.TextEncodingStrategy.get_strategy()

            assert isinstance(tokenize_strategy_cached, strategy_sbxl.SBXLTokenizeStrategy)
            assert isinstance(text_encoding_strategy_cached, strategy_sbxl.SBXLTextEncodingStrategy)

            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    prompt_candidates = [
                        prompt_dict.get("prompt", ""),
                        prompt_dict.get("negative_prompt", ""),
                    ]
                    for idx, prompt_text in enumerate(prompt_candidates):
                        if prompt_text in sample_prompts_te_outputs:
                            continue
                        logger.info(f"cache Text Encoder outputs for prompt: {prompt_text}")
                        tokens_and_masks = tokenize_strategy_cached.tokenize(prompt_text, is_negative=(idx == 1))
                        sample_prompts_te_outputs[prompt_text] = text_encoding_strategy_cached.encode_tokens(
                            tokenize_strategy_cached,
                            [text_encoder],
                            tokens_and_masks,
                        )
        
        # Move text encoder back to CPU to save memory
        text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)
    
    # Prepare training models and optimizers
    logger.info("Preparing training models")
    training_models = []
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)
    
    # Move UNet to device
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(True)
    training_models.append(unet)
    
    # Prepare optimizer
    logger.info("Preparing optimizer")
    trainable_params = list(unet.parameters())
    
    if train_text_encoder:
        logger.info("Text encoder will be trained")
        text_encoder.requires_grad_(True)
        trainable_params += list(text_encoder.parameters())
        training_models.append(text_encoder)
        # Move text encoder to device if training
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    else:
        logger.info("Text encoder will NOT be trained")
        # Keep text encoder on CPU if not training and caching outputs
        if not args.cache_text_encoder_outputs:
            text_encoder.to(accelerator.device, dtype=weight_dtype)
            text_encoder.eval()
    
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)
    
    # Prepare data loader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()
    
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )
    
    # Calculate training steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        logger.info(f"Override max_train_steps to {args.max_train_steps} for {args.max_train_epochs} epochs")
    
    train_dataset_group.set_max_train_steps(args.max_train_steps)
    
    # Prepare learning rate scheduler
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    
    # Prepare accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Fix BatchNorm weights for BF16 compatibility after accelerator.prepare
    if hasattr(args, 'full_bf16') and args.full_bf16:
        logger.info("Converting BatchNorm weights to float32 after accelerator.prepare")
        for module in unet.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()
                module.running_mean.data = module.running_mean.data.float()
                module.running_var.data = module.running_var.data.float()
    
    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if hasattr(args, 'full_fp16') and args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        if text_encoder is not None:
            text_encoder.to(weight_dtype)
    elif hasattr(args, 'full_bf16') and args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        if text_encoder is not None:
            text_encoder.to(weight_dtype)
        
        # Fix BatchNorm weights for BF16 compatibility after full_bf16 conversion
        for module in unet.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()
                module.running_mean.data = module.running_mean.data.float()
                module.running_var.data = module.running_var.data.float()
    
    # Resume training if specified
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)
    
    # Calculate epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Noise scheduler for flow matching
    from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, 
        shift=getattr(args, 'discrete_flow_shift', 3.0)
    )
    
    # Training loop
    logger.info("Starting Training")
    logger.info(f"  Num examples: {train_dataset_group.num_train_images}")
    logger.info(f"  Num batches per epoch: {len(train_dataloader)}")
    logger.info(f"  Num epochs: {num_train_epochs}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps: {args.max_train_steps}")
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"  Batch size per device: {args.train_batch_size}")
    logger.info(f"  Total train batch size: {total_batch_size}")
    
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    
    # Initialize trackers
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            if toml is None:
                raise ImportError("toml package is required when --log_tracker_config is used")
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "sbxl_training" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )
    
    loss_recorder = train_util.LossRecorder()

    sampling_text_encoders = None
    if train_text_encoder or not args.cache_text_encoder_outputs:
        sampling_text_encoders = [text_encoder]

    if ae is not None:
        sbxl_train_util.sample_images(
            accelerator=accelerator,
            args=args,
            epoch=0,
            global_step=global_step,
            unet=unet,
            vae=ae,
            text_encoders=sampling_text_encoders,
            sample_prompts_te_outputs=sample_prompts_te_outputs,
        )
    
    def build_save_metadata(epoch_no: Optional[int], step_no: Optional[int]) -> dict:
        metadata = {
            "ss_model_type": "sbxl",
            "ss_discrete_flow_shift": args.discrete_flow_shift,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_batch_size_per_device": args.train_batch_size,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        sample_guidance = getattr(args, "sample_guidance_scale", None)
        if sample_guidance is not None:
            metadata["ss_sample_guidance_scale"] = sample_guidance
        training_comment = getattr(args, "training_comment", None)
        if training_comment:
            metadata["ss_training_comment"] = training_comment
        if args.seed is not None:
            metadata["ss_seed"] = args.seed
        metadata["ss_git_revision"] = train_util.get_git_revision_hash()
        if epoch_no is not None:
            metadata["ss_epoch"] = epoch_no
        if step_no is not None:
            metadata["ss_steps"] = step_no
        return metadata

    def save_checkpoint(ckpt_name: str, epoch_no: Optional[int], step_no: Optional[int]):
        save_path = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        sbxl_utils.save_sbxl_unet(unwrapped_unet, save_path, dtype=save_dtype, metadata=build_save_metadata(epoch_no, step_no))
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, save_path, "/" + ckpt_name)
        
        # Save training state if requested
        if args.save_state:
            state_name = ckpt_name.replace('.sft', '_state').replace('.safetensors', '_state')
            state_dir = os.path.join(args.output_dir, state_name)
            logger.info(f"Saving training state to {state_dir}")
            accelerator.save_state(state_dir)
            if args.save_state_to_huggingface and args.huggingface_repo_id is not None:
                huggingface_util.upload(args, state_dir, "/" + state_name)

    for epoch in range(num_train_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        
        for m in training_models:
            m.train()
        
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            
            with accelerator.accumulate(*training_models):
                # Get latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    if ae is None:
                        raise ValueError(
                            "Latents not provided in batch and VAE not available. Pass --vae or enable --cache_latents."
                        )
                    with torch.no_grad():
                        images = batch["images"].to(ae.device, dtype=ae.dtype)
                        latents = ae.encode(images)
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Get text encoder outputs
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list")
                if text_encoder_outputs_list is not None and not train_text_encoder:
                    txt_emb = text_encoder_outputs_list[0].to(accelerator.device, dtype=weight_dtype)
                    txt_mask = None
                    if len(text_encoder_outputs_list) > 2 and text_encoder_outputs_list[2] is not None:
                        txt_mask = text_encoder_outputs_list[2].to(accelerator.device)
                else:
                    tokens = batch.get("input_ids_list")
                    if tokens is None:
                        raise ValueError("Input IDs missing from batch; cannot encode text.")
                    token_tensors = [ids.to(accelerator.device) for ids in tokens]
                    txt_emb, _, txt_mask = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        [text_encoder],
                        tuple(token_tensors),
                    )
                    txt_emb = txt_emb.to(accelerator.device, dtype=weight_dtype)
                    if txt_mask is not None:
                        txt_mask = txt_mask.to(accelerator.device)
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                
                noisy_model_input, timesteps, sigmas = sbxl_train_util.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
                )
                
                # Predict velocity
                model_pred = unet(
                    sample=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=txt_emb,
                    attention_mask=txt_mask,
                    return_dict=False,
                )[0]
                
                model_pred, weighting = sbxl_train_util.apply_model_prediction_type(
                    args, model_pred, noisy_model_input, sigmas
                )

                # Calculate target (velocity)
                target = noise - latents

                timesteps_indices = timesteps.round().clamp(min=0, max=noise_scheduler.config.num_train_timesteps - 1)
                timesteps_indices = timesteps_indices.to(torch.long)
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps_indices, noise_scheduler)
                loss = train_util.conditional_loss(
                    model_pred.float(),
                    target.float(),
                    args.loss_type,
                    "none",
                    huber_c,
                )
                if weighting is not None:
                    loss = loss * weighting

                if args.masked_loss or ("network_multipliers" in batch and batch["network_multipliers"] is not None):
                    loss = apply_masked_loss(loss, batch)

                loss = loss.mean(dim=(1, 2, 3))

                loss_weights = batch.get("loss_weights")
                if loss_weights is not None:
                    loss = loss * loss_weights.to(loss.device, dtype=loss.dtype)

                loss = loss.mean()
                
                accelerator.backward(loss)
                
                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = trainable_params
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                
                # Step learning rate scheduler when gradients are synced
                if accelerator.sync_gradients and args.fused_backward_pass:
                    lr_scheduler.step()
            
            # Record loss
            if args.logging_dir is not None:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if args.log_with:
                    accelerator.log(logs, step=global_step)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if ae is not None:
                    sbxl_train_util.sample_images(
                        accelerator=accelerator,
                        args=args,
                        epoch=None,
                        global_step=global_step,
                        unet=unet,
                        vae=ae,
                        text_encoders=sampling_text_encoders,
                        sample_prompts_te_outputs=sample_prompts_te_outputs,
                    )

                # Save checkpoint
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_name = train_util.get_step_ckpt_name(args, ckpt_ext, global_step)
                        logger.info(f"Saving checkpoint to {os.path.join(args.output_dir, ckpt_name)}")
                        save_checkpoint(ckpt_name, epoch + 1, global_step)

                        remove_step = train_util.get_remove_step_no(args, global_step)
                        if remove_step is not None:
                            old_ckpt = train_util.get_step_ckpt_name(args, ckpt_ext, remove_step)
                            old_path = os.path.join(args.output_dir, old_ckpt)
                            if os.path.exists(old_path):
                                logger.info(f"Removing old checkpoint {old_path}")
                                os.remove(old_path)
                        
                        # Save state stepwise if requested
                        if args.save_last_n_steps_state is not None:
                            import os
                            model_name = train_util.default_if_none(args.output_name, "step")
                            state_dir = os.path.join(args.output_dir, f"{model_name}-{global_step}-state")
                            logger.info(f"Saving training state at step {global_step} to {state_dir}")
                            accelerator.save_state(state_dir)
                            if args.save_state_to_huggingface and args.huggingface_repo_id is not None:
                                huggingface_util.upload(args, state_dir, f"/{model_name}-{global_step}-state")
                            
                            # Remove old states
                            last_n_steps = args.save_last_n_steps_state
                            remove_step_no = global_step - last_n_steps - 1
                            remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps) if args.save_every_n_steps > 0 else remove_step_no
                            if remove_step_no > 0:
                                old_state_dir = os.path.join(args.output_dir, f"{model_name}-{remove_step_no}-state")
                                if os.path.exists(old_state_dir):
                                    logger.info(f"Removing old state: {old_state_dir}")
                                    import shutil
                                    shutil.rmtree(old_state_dir)
            
            if global_step >= args.max_train_steps:
                break
        
        # Save epoch checkpoint
        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_name = train_util.get_epoch_ckpt_name(args, ckpt_ext, epoch + 1)
                logger.info(f"Saving epoch checkpoint to {os.path.join(args.output_dir, ckpt_name)}")
                save_checkpoint(ckpt_name, epoch + 1, global_step)

                remove_epoch = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch is not None:
                    old_ckpt = train_util.get_epoch_ckpt_name(args, ckpt_ext, remove_epoch)
                    old_path = os.path.join(args.output_dir, old_ckpt)
                    if os.path.exists(old_path):
                        logger.info(f"Removing old checkpoint {old_path}")
                        os.remove(old_path)
                
                # Save state on epoch end if requested
                if args.save_last_n_epochs_state is not None:
                    import os
                    model_name = train_util.default_if_none(args.output_name, "epoch")
                    state_dir = os.path.join(args.output_dir, f"{model_name}-{epoch + 1}-state")
                    logger.info(f"Saving training state at epoch {epoch + 1} to {state_dir}")
                    accelerator.save_state(state_dir)
                    if args.save_state_to_huggingface and args.huggingface_repo_id is not None:
                        huggingface_util.upload(args, state_dir, f"/{model_name}-{epoch + 1}-state")
                    
                    # Remove old states
                    last_n_epochs = args.save_last_n_epochs_state
                    remove_epoch_no = (epoch + 1) - args.save_every_n_epochs * last_n_epochs
                    if remove_epoch_no > 0:
                        old_state_dir = os.path.join(args.output_dir, f"{model_name}-{remove_epoch_no}-state")
                        if os.path.exists(old_state_dir):
                            logger.info(f"Removing old state: {old_state_dir}")
                            import shutil
                            shutil.rmtree(old_state_dir)
        
        if global_step >= args.max_train_steps:
            break

        if ae is not None:
            sbxl_train_util.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=epoch + 1,
                global_step=global_step,
                unet=unet,
                vae=ae,
                text_encoders=sampling_text_encoders,
                sample_prompts_te_outputs=sample_prompts_te_outputs,
            )
    
    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, ckpt_ext)
        logger.info(f"Saving final checkpoint to {os.path.join(args.output_dir, ckpt_name)}")
        save_checkpoint(ckpt_name, num_train_epochs, global_step)
    
    # Save final training state if requested
    if args.save_state_on_train_end:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            import os
            model_name = train_util.default_if_none(args.output_name, "last")
            state_dir = os.path.join(args.output_dir, f"{model_name}_state")
            logger.info(f"Saving final training state to {state_dir}")
            accelerator.save_state(state_dir)
            if args.save_state_to_huggingface and args.huggingface_repo_id is not None:
                huggingface_util.upload(args, state_dir, f"/{model_name}_state")
    
    accelerator.end_training()
    logger.info("Training complete!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SBXL full model training")
    
    # Add base training arguments
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_masked_loss_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    add_logging_arguments(parser)
    
    # SBXL specific arguments
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to Sakiko text encoder (default: from HuggingFace)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt for text encoder",
    )
    parser.add_argument(
        "--sbxl_params_file",
        type=str,
        default=None,
        help="Path to JSON/TOML file with SBXL architecture parameters",
    )
    parser.add_argument(
        "--sbxl_param",
        type=str,
        nargs="*",
        default=None,
        help="Override SBXL architecture parameters (key=value)",
    )
    parser.add_argument(
        "--sbxl_vae",
        type=str,
        required=True,
        help="Path to SBXL VAE model",
    )
    
    # Modify vae argument to be required for SBXL
    for action in parser._actions:
        if action.dest == 'vae':
            action.required = False  # Disable the original vae requirement
            break
    
    # Training parameters
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Train text encoder along with UNet",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Discrete flow shift for noise scheduler",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use Flash Attention kernels for SBXL cross attention",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs",
        action="store_true",
        help="cache text encoder outputs / text encoderの出力をキャッシュする",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default=None,
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", None],
        help="Weighting scheme for loss",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean for logit normal weighting",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std for logit normal weighting",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale for mode weighting",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "sigmoid", "shift"],
        help="Timestep sampling method",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale for sigmoid timestep sampling",
    )
    parser.add_argument(
        "--model_prediction_type",
        type=str,
        default="raw",
        choices=["raw", "additive", "sigma_scaled"],
        help="Model prediction type",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="Batch size for text encoder output caching",
    )
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="Offload checkpointing to CPU",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead / 代わりに 'skip_cache_check' を使用してください",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=28,
        help="Number of sampling steps for validation images",
    )
    parser.add_argument(
        "--sample_guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for validation images",
    )
    parser.add_argument(
        "--sample_height",
        type=int,
        default=None,
        help="Height for validation images (default: same as resolution)",
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=None,
        help="Width for validation images (default: same as resolution)",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=None,
        help="Batch size for validation sampling (default: train batch size)",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="Comment to add to training metadata",
    )
    
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    
    train(args)


if __name__ == "__main__":
    main()

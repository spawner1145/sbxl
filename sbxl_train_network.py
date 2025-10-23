# SBXL Network (LoRA) Training Script
# Based on flux_train_network.py

import argparse
import copy
import json
import math
import os
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_train_utils,
    sbxl_models,
    sbxl_train_util,
    sbxl_utils,
    strategy_base,
    strategy_sbxl,
    train_util,
)
from library.custom_train_functions import apply_masked_loss
from library.utils import setup_logging
import networks.lora_sbxl as lora_sbxl

setup_logging()
import logging
logger = logging.getLogger(__name__)


class SBXLNetworkTrainer(train_network.NetworkTrainer):
    """Trainer for SBXL LoRA/network training"""
    
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
    
    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        
        # Check cache settings
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, enabling cache_text_encoder_outputs"
            )
            args.cache_text_encoder_outputs = True
        
        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "When caching text encoder outputs, caption_dropout_rate, shuffle_caption, etc. cannot be used"
        
        # Prepare text encoder training flag
        self.train_text_encoder = not args.network_train_unet_only
        
        # Verify bucket resolution steps
        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)
    
    def load_target_model(self, args, weight_dtype, accelerator):
        """Load SBXL UNet, text encoder, and VAE"""
        
        # Resolve SBXL architecture parameters
        sbxl_preset_overrides, sbxl_cli_overrides = sbxl_utils.get_sbxl_params_from_args(
            getattr(args, "sbxl_params_file", None),
            getattr(args, "sbxl_param", None),
        )

        # Load UNet
        logger.info(f"Loading SBXL UNet from {args.pretrained_model_name_or_path}")
        unet = sbxl_utils.load_sbxl_unet(
            args.pretrained_model_name_or_path,
            dtype=weight_dtype,
            device="cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
            preset_overrides=sbxl_preset_overrides,
            cli_overrides=sbxl_cli_overrides,
        )
        args.sbxl_resolved_params = unet.params.copy()
        attention_backend = sbxl_utils.select_attention_backend_from_args(args)
        sbxl_utils.configure_attention_backend(unet, attention_backend)
        
        # Fix BatchNorm weights for BF16 compatibility
        if weight_dtype == torch.bfloat16:
            logger.info("Converting BatchNorm weights to float32 for BF16 compatibility")
            for module in unet.modules():
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    module.weight.data = module.weight.data.float()
                    module.bias.data = module.bias.data.float()
                    module.running_mean.data = module.running_mean.data.float()
                    module.running_var.data = module.running_var.data.float()
        
        # Load text encoder
        logger.info("Loading Sakiko text encoder")
        text_encoder = sbxl_utils.load_text_encoder(
            args.text_encoder_path if hasattr(args, 'text_encoder_path') else None,
            dtype=weight_dtype,
            device="cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
            cache_dir=args.tokenizer_cache_dir if hasattr(args, 'tokenizer_cache_dir') else None,
        )
        text_encoder.eval()
        
        # Load VAE
        logger.info(f"Loading Flux VAE from {args.sbxl_vae}")
        vae = sbxl_utils.load_vae(
            args.sbxl_vae,
            dtype=weight_dtype,
            device="cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
        )
        
        model_version = sbxl_utils.MODEL_VERSION_SBXL_V1
        return model_version, [text_encoder], vae, unet
    
    def get_tokenize_strategy(self, args):
        """Get tokenization strategy"""
        max_length = getattr(args, 'max_token_length', 384)
        system_prompt = getattr(args, 'system_prompt', None)
        
        logger.info(f"Using Sakiko tokenizer with max_length={max_length}")
        if system_prompt:
            logger.info(f"System prompt: {system_prompt[:50]}...")
        
        return strategy_sbxl.SBXLTokenizeStrategy(
            max_length=max_length,
            tokenizer_path=getattr(args, 'text_encoder_path', None),
            tokenizer_cache_dir=getattr(args, 'tokenizer_cache_dir', None),
            system_prompt=system_prompt,
        )
    
    def get_tokenizers(self, tokenize_strategy: strategy_sbxl.SBXLTokenizeStrategy):
        """Get tokenizers"""
        return [tokenize_strategy.tokenizer]
    
    def get_latents_caching_strategy(self, args):
        """Get latents caching strategy"""
        latents_caching_strategy = strategy_sbxl.SBXLLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            False,
        )
        return latents_caching_strategy
    
    def get_text_encoding_strategy(self, args):
        """Get text encoding strategy"""
        return strategy_sbxl.SBXLTextEncodingStrategy()
    
    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """Post-process network after creation"""
        # Nothing special to do for SBXL
        pass
    
    def get_text_encoder_outputs_caching_strategy(self, args):
        """Get text encoder outputs caching strategy"""
        if args.cache_text_encoder_outputs:
            # if the text encoder is trained, we need tokenization, so is_partial is True
            return strategy_sbxl.SBXLTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size if hasattr(args, 'text_encoder_batch_size') else args.vae_batch_size,
                args.skip_cache_check if hasattr(args, 'skip_cache_check') else False,
                is_partial=self.train_text_encoder,
            )
        else:
            return None
    
    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        """Cache text encoder outputs if needed"""
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # Reduce memory consumption
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy: strategy_sbxl.SBXLTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_sbxl.SBXLTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
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
                            tokens_and_masks = tokenize_strategy.tokenize(prompt_text, is_negative=(idx == 1))
                            sample_prompts_te_outputs[prompt_text] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu if not training text encoder
            if not self.train_text_encoder:
                logger.info("move text encoder back to cpu")
                text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Get text encoder output every time, so keep it on GPU
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
    
    def create_network(
        self,
        multiplier: float,
        network_dim: int,
        network_alpha: float,
        vae: Any,
        text_encoder: Any,
        unet: sbxl_models.SBXLUNet2DConditionModel,
        **kwargs,
    ):
        """Create LoRA network"""
        network = lora_sbxl.SBXLLoRANetwork(
            unet=unet,
            text_encoder=text_encoder[0] if self.train_text_encoder else None,
            multiplier=multiplier,
            lora_dim=network_dim,
            alpha=network_alpha,
            dropout=kwargs.get("dropout", None),
            rank_dropout=kwargs.get("rank_dropout", None),
            module_dropout=kwargs.get("module_dropout", None),
            target_modules=kwargs.get("network_module", None),
        )
        
        return network
    
    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        """Prepare UNet with accelerator and fix BatchNorm weights for BF16 compatibility"""
        unet = accelerator.prepare(unet)
        
        # Fix BatchNorm weights for BF16 compatibility after accelerator.prepare
        if hasattr(args, 'full_bf16') and args.full_bf16:
            for module in unet.modules():
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    module.weight.data = module.weight.data.float()
                    module.bias.data = module.bias.data.float()
                    module.running_mean.data = module.running_mean.data.float()
                    module.running_var.data = module.running_var.data.float()
        
        return unet
    
    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        """Sample images during training"""
        sbxl_train_util.sample_images(
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            global_step=global_step,
            unet=unet,
            vae=vae,
            text_encoders=self.get_models_for_text_encoding(args, accelerator, [text_encoder]),
            sample_prompts_te_outputs=self.sample_prompts_te_outputs,
        )
    
    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """Get models for text encoding"""
        if args.cache_text_encoder_outputs:
            if self.train_text_encoder:
                return text_encoders  # text encoder is needed for encoding because it's being trained
            else:
                return None  # no text encoders are needed for encoding because outputs are cached
        else:
            return text_encoders  # text encoder is needed for encoding
    
    def get_text_encoders_train_flags(self, args, text_encoders):
        """Get training flags for text encoders"""
        return [self.train_text_encoder]
    
    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ):
        """
        Process a batch for SBXL training with full_fp16 support
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device)
            else:
                # Encode images to latents
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = torch.nan_to_num(latents, 0, out=latents)

            latents = self.shift_scale_latents(args, latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids,
                )
                if args.full_fp16 or args.full_bf16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
        if weighting is not None:
            loss = loss * weighting
        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        return loss.mean()
    
    def get_noise_scheduler(self, args, device):
        """Get noise scheduler for SBXL (Euler discrete flow matching)"""
        from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, 
            shift=getattr(args, 'discrete_flow_shift', 3.0)
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler
    
    def encode_images_to_latents(self, args, vae, images):
        """Encode images to latents using VAE"""
        # Ensure VAE is on the correct device and images are in correct dtype
        vae.to(images.device)
        return vae.encode(images.to(vae.dtype))
    
    def shift_scale_latents(self, args, latents):
        """Shift and scale latents - SBXL uses Flux VAE which handles scaling internally"""
        return latents
    
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: sbxl_models.SBXLUNet2DConditionModel,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        """Get noise prediction and target for SBXL flow matching"""
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = sbxl_train_util.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # get text embeddings and masks
        if isinstance(text_encoder_conds, (list, tuple)) and len(text_encoder_conds) >= 3:
            txt_emb, _, txt_mask = text_encoder_conds
        else:
            txt_emb, txt_mask = text_encoder_conds

        # Predict the noise
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            if txt_emb is not None and txt_emb.dtype.is_floating_point:
                txt_emb.requires_grad_(True)
            if txt_mask is not None and txt_mask.dtype.is_floating_point:
                txt_mask.requires_grad_(True)

        def call_unet(img, txt_emb, txt_mask, timesteps):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = unet(
                    sample=img,
                    timestep=timesteps,
                    encoder_hidden_states=txt_emb,
                    attention_mask=txt_mask,
                    return_dict=False,
                )[0]
            return model_pred

        model_pred = call_unet(
            img=noisy_model_input,
            txt_emb=txt_emb,
            txt_mask=txt_mask,
            timesteps=timesteps,
        )

        # apply model prediction type (if any)
        model_pred, weighting = sbxl_train_util.apply_model_prediction_type(
            args, model_pred, noisy_model_input, sigmas
        )

        # flow matching loss: velocity = noise - latents
        target = noise - latents

        # differential output preservation (if specified)
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad():
                    model_pred_prior = call_unet(
                        img=noisy_model_input[diff_output_pr_indices],
                        txt_emb=txt_emb[diff_output_pr_indices],
                        txt_mask=txt_mask[diff_output_pr_indices] if txt_mask is not None else None,
                        timesteps=timesteps[diff_output_pr_indices],
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step

                if weighting is not None:
                    model_pred_prior, _ = sbxl_train_util.apply_model_prediction_type(
                        args,
                        model_pred_prior,
                        noisy_model_input[diff_output_pr_indices],
                        sigmas[diff_output_pr_indices] if sigmas is not None else None,
                    )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        timesteps_for_loss = timesteps.round().clamp(
            min=0, max=noise_scheduler.config.num_train_timesteps - 1
        ).to(torch.long)

        return model_pred, target, timesteps_for_loss, weighting
    
    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """Post process loss - SBXL doesn't need special processing"""
        return loss
    
    def get_sai_model_spec(self, args):
        """Get SAI model spec for SBXL"""
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")
    
    def update_metadata(self, metadata, args):
        """Update metadata with SBXL specific info"""
        metadata["ss_model_type"] = "sbxl"
        if hasattr(args, 'weighting_scheme'):
            metadata["ss_weighting_scheme"] = args.weighting_scheme
        if hasattr(args, 'timestep_sampling'):
            metadata["ss_timestep_sampling"] = args.timestep_sampling
        if hasattr(args, 'discrete_flow_shift'):
            metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        if hasattr(args, 'system_prompt'):
            metadata["ss_system_prompt"] = args.system_prompt
        sample_guidance = getattr(args, 'sample_guidance_scale', None)
        if sample_guidance is not None:
            metadata["ss_sample_guidance_scale"] = sample_guidance
        resolved_params = getattr(args, "sbxl_resolved_params", None)
        if isinstance(resolved_params, sbxl_models.SBXLParams):
            metadata["ss_sbxl_params"] = json.dumps(resolved_params.to_serializable_dict())
    
    def is_text_encoder_not_needed_for_training(self, args):
        """Check if text encoder is not needed for training"""
        return args.cache_text_encoder_outputs and not self.train_text_encoder
    
    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """Prepare text encoder for gradient checkpointing - SBXL specific"""
        # For SBXL's Sakiko text encoder, check if it has the expected structure
        if hasattr(text_encoder, 'text_model') and hasattr(text_encoder.text_model, 'embeddings'):
            text_encoder.text_model.embeddings.requires_grad_(True)
        else:
            # For other structures, try to find embeddings
            if hasattr(text_encoder, 'embeddings'):
                text_encoder.embeddings.requires_grad_(True)
            else:
                logger.warning("Could not find embeddings in text encoder for gradient checkpointing workaround")
    
    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """Prepare text encoder for fp8 - SBXL specific"""
        # For SBXL's Sakiko text encoder, check if it has the expected structure
        if hasattr(text_encoder, 'text_model') and hasattr(text_encoder.text_model, 'embeddings'):
            text_encoder.text_model.embeddings.to(dtype=te_weight_dtype)
        else:
            # For other structures, try to find embeddings
            if hasattr(text_encoder, 'embeddings'):
                text_encoder.embeddings.to(dtype=te_weight_dtype)
            else:
                logger.warning("Could not find embeddings in text encoder for fp8 preparation")


def setup_parser() -> argparse.ArgumentParser:
    import os
    parser = train_network.setup_parser()
    # train_util.add_dit_training_arguments(parser)  # SBXL has its own arguments
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
        help="System prompt for text encoder (default: standard image generation prompt)",
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
    # Check if vae argument already exists
    try:
        parser.add_argument(
            "--sbxl_vae",
            type=str,
            required=True,
            help="Path to Flux VAE checkpoint",
        )
    except argparse.ArgumentError:
        # Argument already exists, modify it to be required for SBXL
        for action in parser._actions:
            if action.dest == 'vae':
                action.required = True
                break
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
        help="Weighting scheme for loss calculation",
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
    
    # Training arguments are already added by train_network.setup_parser()
    # train_util.add_training_arguments(parser, support_dreambooth=False)
    
    # Sample arguments
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
    
    # tokenizer_cache_dir is already added by train_util.add_sd_models_arguments()
    # parser.add_argument(
    #     "--tokenizer_cache_dir",
    #     type=str,
    #     default=None,
    #     help="Cache directory for tokenizer",
    # )
    
    return parser


def main():
    parser = setup_parser()
    
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    
    trainer = SBXLNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()


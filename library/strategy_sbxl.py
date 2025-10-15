# SBXL Training Strategies
# Tokenization, Text Encoding, and Latents Caching strategies for SBXL

import glob
import os
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from library import train_util, sbxl_utils
from library.strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
)
from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)


SAKIKO_TEXT_ENCODER_ID = "SakikoLab/Sakiko-Prompt-Gen-v1.0"
SAKIKO_MAX_TOKEN_LENGTH = 384


class SBXLTokenizeStrategy(TokenizeStrategy):
    """Tokenization strategy for SBXL using Sakiko tokenizer"""
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        tokenizer_cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize tokenization strategy
        
        Args:
            max_length: Maximum token length (default: 384)
            tokenizer_cache_dir: Cache directory for tokenizer
            system_prompt: System prompt to prepend to all prompts
        """
        resolved_tokenizer_id = tokenizer_path
        if resolved_tokenizer_id:
            if os.path.isfile(resolved_tokenizer_id):
                resolved_tokenizer_id = os.path.dirname(resolved_tokenizer_id)
        else:
            resolved_tokenizer_id = SAKIKO_TEXT_ENCODER_ID

        logger.info(f"Loading SBXL tokenizer from {resolved_tokenizer_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_id,
            cache_dir=tokenizer_cache_dir,
        )
        self.tokenizer.padding_side = "right"
        
        if max_length is None:
            self.max_length = SAKIKO_MAX_TOKEN_LENGTH
        else:
            self.max_length = max_length
        
        # System prompt support
        if system_prompt is None:
            system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
        self.system_prompt = system_prompt
        
        # Check if tokenizer has chat template
        self.use_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        logger.info(f"Initialized SBXL tokenizer with max_length={self.max_length}")
        logger.info(f"System prompt: {self.system_prompt[:50]}...")
        logger.info(f"Chat template support: {self.use_chat_template}")
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        is_negative: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text with system prompt support
        
        Args:
            text: Text or list of texts to tokenize
            is_negative: Whether this is a negative prompt (no system prompt for negative)
        
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        text = [text] if isinstance(text, str) else text
        
        # Apply system prompt for positive prompts
        if not is_negative:
            processed_text = []
            for t in text:
                if self.use_chat_template:
                    try:
                        messages = [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": t}
                        ]
                        full_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=False,
                            tokenize=False
                        )
                    except Exception as e:
                        logger.warning(f"chat_template failed, using text concatenation: {e}")
                        full_prompt = f'{self.system_prompt} <Prompt Start> {t}'
                else:
                    full_prompt = f'{self.system_prompt} <Prompt Start> {t}'
                processed_text.append(full_prompt)
        else:
            processed_text = text
        
        encodings = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        
        return (encodings.input_ids, encodings.attention_mask)
    
    def tokenize_with_weights(
        self,
        text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Tokenize text with weights (weights not supported yet, returns uniform weights)
        
        Args:
            text: Text or list of texts to tokenize
        
        Returns:
            Tuple of (input_ids, attention_mask, weights)
        """
        tokens, attention_masks = self.tokenize(text)
        # Uniform weights for now (no prompt weighting syntax support yet)
        weights = [torch.ones_like(t, dtype=torch.float32) for t in tokens]
        return tokens, attention_masks, weights


class SBXLTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for SBXL"""
    
    def __init__(self) -> None:
        super().__init__()
    
    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode tokens using Sakiko text encoder
        
        Args:
            tokenize_strategy: Tokenization strategy
            models: List containing text encoder model
            tokens: Tuple of (input_ids, attention_mask)
        
        Returns:
            Tuple of (hidden_states, attention_mask)
        """
        text_encoder = models[0]
        input_ids, attention_mask = tokens
        
        # Encode text
        input_ids = input_ids.to(text_encoder.device)
        attention_mask = attention_mask.to(text_encoder.device)

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

        return hidden_states, input_ids, attention_mask


class SBXLTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Text encoder outputs caching strategy for SBXL"""
    
    SBXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_sbxl_te.npz"
    
    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
    
    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        """Get path for cached outputs"""
        return (
            os.path.splitext(image_abs_path)[0] + self.SBXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        )
    
    def is_disk_cached_outputs_expected(self, image_abs_path: str) -> bool:
        """Check if cached outputs exist"""
        return os.path.exists(self.get_outputs_npz_path(image_abs_path))
    
    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List[train_util.ImageInfo],
    ):
        """Cache batch of text encoder outputs"""
        text_encoder = models[0]

        captions = [info.caption for info in batch]
        tokens = tokenize_strategy.tokenize(captions)

        with torch.no_grad():
            hidden_states, _, encoded_attention_mask = text_encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )

        hidden_states = hidden_states.float().cpu()
        input_ids_tensor, attention_mask_tensor = tokens
        input_ids = input_ids_tensor.cpu()
        attention_mask_outputs = encoded_attention_mask.cpu() if encoded_attention_mask is not None else attention_mask_tensor.cpu()

        for i, info in enumerate(batch):
            hidden_state_i = hidden_states[i].numpy()
            input_ids_i = input_ids[i].numpy()
            attention_mask_i = attention_mask_outputs[i].numpy()

            if self.cache_to_disk:
                assert (
                    info.text_encoder_outputs_npz is not None
                ), f"Text encoder cache path missing for image {info.image_key}"
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_states=hidden_state_i,
                    input_ids=input_ids_i,
                    attention_mask=attention_mask_i,
                )
            else:
                info.text_encoder_outputs = [
                    hidden_state_i,
                    input_ids_i,
                    attention_mask_i,
                ]
    
    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        """Load cached outputs from disk"""
        data = np.load(npz_path)
        return [data["hidden_states"], data["input_ids"], data["attention_mask"]]
    
    def _default_batch_processor(
        self,
        batch: List[train_util.ImageInfo],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Default batch processor for loading cached outputs"""
        hidden_states_list = []
        attention_mask_list = []
        input_ids_list = []

        for info in batch:
            if self.cache_to_disk:
                outputs = self.load_outputs_npz(info.text_encoder_outputs_npz)
            else:
                outputs = info.text_encoder_outputs

            hidden_state, input_ids, attn_mask = outputs
            hidden_states_list.append(torch.from_numpy(hidden_state))
            input_ids_list.append(torch.from_numpy(input_ids))
            attention_mask_list.append(torch.from_numpy(attn_mask))

        hidden_states = torch.stack(hidden_states_list, dim=0)
        attention_masks = torch.stack(attention_mask_list, dim=0)
        input_ids = torch.stack(input_ids_list, dim=0)

        return hidden_states, input_ids, attention_masks


class SBXLLatentsCachingStrategy(LatentsCachingStrategy):
    """Latents caching strategy for SBXL (uses Flux VAE)"""
    
    SBXL_LATENTS_NPZ_SUFFIX = "_sbxl_latents.npz"
    
    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
    
    @property
    def cache_suffix(self) -> str:
        return self.SBXL_LATENTS_NPZ_SUFFIX
    
    def get_image_size_from_disk_cache_path(self, absolute_path: str, npz_path: str) -> Tuple[Optional[int], Optional[int]]:
        # SBXL uses fixed resolution, so we don't encode size in filename
        return None, None
    
    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=False)
    
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=False
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
    
    def get_latents_npz_path(self, image_abs_path: str, image_size: Tuple[int, int]) -> str:
        """Get path for cached latents"""
        return os.path.splitext(image_abs_path)[0] + self.SBXL_LATENTS_NPZ_SUFFIX
    
    def _default_cache_latents(
        self,
        vae,
        image_infos: List,
    ):
        """Cache latents for images"""
        from library import train_util
        
        img_tensor = self._load_images_for_caching(image_infos)
        img_tensor = img_tensor.to(vae.device, dtype=vae.dtype)
        
        # Encode with Flux VAE (already scaled internally)
        with torch.no_grad():
            latents = vae.encode(img_tensor)
        
        # Cache each latent
        for i, image_info in enumerate(image_infos):
            latent = latents[i : i + 1].cpu().numpy()
            
            if self.cache_to_disk:
                npz_path = self.get_latents_npz_path(image_info.absolute_path)
                np.savez(npz_path, latents=latent)
            else:
                self.cached_latents[image_info.absolute_path] = latent
    
    def _load_images_for_caching(self, image_infos: List) -> torch.Tensor:
        """Load images for caching"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        images = []
        for image_info in image_infos:
            img = Image.open(image_info.absolute_path).convert("RGB")
            img = img.resize((image_info.bucket_reso[0], image_info.bucket_reso[1]), Image.LANCZOS)
            
            # Convert to tensor and normalize
            img_tensor = transforms.ToTensor()(img)
            img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            images.append(img_tensor)
        
        return torch.stack(images)
    
    def load_latents_npz(self, npz_path: str) -> np.ndarray:
        """Load cached latents from disk"""
        data = np.load(npz_path)
        return data["latents"]
    
    def _default_batch_processor(
        self,
        batch: List[Tuple[Any, Optional[str], Optional[str]]],
    ) -> torch.Tensor:
        """Default batch processor for loading cached latents"""
        latents_list = []
        
        for image_info, _, _ in batch:
            image_abs_path = image_info.absolute_path
            
            if self.cache_to_disk:
                npz_path = self.get_latents_npz_path(image_abs_path)
                latent = self.load_latents_npz(npz_path)
            else:
                latent = self.cached_latents[image_abs_path]
            
            latents_list.append(torch.from_numpy(latent))
        
        latents = torch.cat(latents_list, dim=0)
        return latents

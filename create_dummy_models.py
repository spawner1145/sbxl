#!/usr/bin/env python3
"""
Create dummy SBXL models for testing
"""

import torch
import os
from library import sbxl_models, sbxl_utils, flux_utils

def create_dummy_sbxl_unet():
    """Create a dummy SBXL UNet model with random weights"""
    print("Creating dummy SBXL UNet with LSNet configuration...")

    # Create SBXL UNet model with LSNet configuration
    params = sbxl_models.SBXLParams.get_lsnet_config()
    model = sbxl_models.SBXLUNet2DConditionModel(params)

    # Initialize with random weights (but same shape as real model)
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.apply(init_weights)

    # Save the model in BF16 for faster inference
    save_path = "dummy_sbxl_unet_bf16.safetensors"
    sbxl_utils.save_sbxl_unet(model, save_path, dtype=torch.bfloat16)
    print(f"Saved dummy SBXL UNet BF16 to {save_path}")

def create_dummy_flux_vae():
    """Create a dummy Flux VAE model"""
    print("Creating dummy Flux VAE...")

    # Import Flux models
    from library import flux_models

    # Create VAE model using Flux parameters
    vae_params = flux_models.configs["dev"].ae_params
    vae = flux_models.AutoEncoder(vae_params)

    # Initialize with random weights
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    vae.apply(init_weights)

    # Save the model
    save_path = "dummy_flux_vae.safetensors"
    state_dict = vae.state_dict()
    from safetensors.torch import save_file
    save_file(state_dict, save_path)
    print(f"Saved dummy Flux VAE to {save_path}")

def create_dummy_text_encoder():
    """Use Sakiko-Prompt-Gen-v1.0 folder instead of creating dummy"""
    print("Using Sakiko-Prompt-Gen-v1.0 folder for text encoder (no dummy needed)")
    # Text encoder will be loaded from Sakiko-Prompt-Gen-v1.0 folder
    pass

if __name__ == "__main__":
    print("Creating dummy models for SBXL testing...")

    try:
        create_dummy_sbxl_unet()
        #create_dummy_flux_vae()
        create_dummy_text_encoder()
        print("✓ All dummy models created successfully!")
    except Exception as e:
        print(f"✗ Error creating dummy models: {e}")
        import traceback
        traceback.print_exc()
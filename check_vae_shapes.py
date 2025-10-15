import torch
from safetensors.torch import load_file

# Load VAE file to check shapes
vae_path = r'D:\code\sbxl\sd-scripts\flux vae.safetensors'
sd = load_file(vae_path)

print('Checking VAE layer shapes:')

# Check some key layers with original HF names
key_layers = [
    'encoder.conv_in.weight',
    'encoder.conv_out.weight', 
    'decoder.conv_in.weight',
    'decoder.conv_out.weight',
    'decoder.up_blocks.0.resnets.0.conv1.weight',
    'decoder.up_blocks.1.resnets.0.conv1.weight', 
    'decoder.up_blocks.2.resnets.0.conv1.weight',
    'decoder.up_blocks.3.resnets.0.conv1.weight',
]

for key in key_layers:
    if key in sd:
        shape = sd[key].shape
        print(f'{key}: {shape}')
    else:
        print(f'{key}: NOT FOUND')

# Check our Flux VAE params
from library import flux_models
ae_params = flux_models.configs["dev"].ae_params
print(f'\nOur Flux VAE params:')
print(f'  ch: {ae_params.ch}')
print(f'  ch_mult: {ae_params.ch_mult}')
print(f'  num_res_blocks: {ae_params.num_res_blocks}')
print(f'  z_channels: {ae_params.z_channels}')

# Calculate expected channel sizes
ch = ae_params.ch
ch_mult = [1] + list(ae_params.ch_mult)  # [1, 1, 2, 4, 4]
channels = [ch * m for m in ch_mult]  # [128, 128, 256, 512, 512]
print(f'Expected channels per level: {channels}')
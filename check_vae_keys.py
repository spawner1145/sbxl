import torch
from safetensors.torch import load_file

# Load VAE file to see the keys
vae_path = r'D:\code\sbxl\sd-scripts\flux vae.safetensors'
sd = load_file(vae_path)

print('VAE file keys (first 20):')
for i, key in enumerate(sorted(sd.keys())):
    if i < 20:
        print(f'  {key}')
    elif i == 20:
        print('  ...')
        break

print(f'\nTotal keys: {len(sd)}')

# Check if there are vae. prefix keys
vae_keys = [k for k in sd.keys() if k.startswith('vae.')]
print(f'Keys with vae. prefix: {len(vae_keys)}')

# Check encoder keys
encoder_keys = [k for k in sd.keys() if k.startswith('encoder.')]
print(f'Keys with encoder. prefix: {len(encoder_keys)}')

# Show some encoder keys
print('\nSome encoder keys:')
for key in sorted(encoder_keys)[:10]:
    print(f'  {key}')

# Show decoder keys
decoder_keys = [k for k in sd.keys() if k.startswith('decoder.')]
print(f'\nKeys with decoder. prefix: {len(decoder_keys)}')

print('\nSome decoder keys:')
for key in sorted(decoder_keys)[:10]:
    print(f'  {key}')
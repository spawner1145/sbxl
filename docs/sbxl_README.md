# SBXL - A New Diffusion Model Architecture

## Overview

SBXL is a new diffusion model architecture that combines:
- **Text Encoder**: SakikoLab/Sakiko-Prompt-Gen-v1.0 (hidden_size: 1024, max_tokens: 384)
- **UNet**: SDXL-based architecture with LSNet components replacing traditional conv layers
- **VAE**: Flux VAE (16 channels, high quality)
- **Sampling**: Flow-based sampling (like Flux)

## Architecture Details

### Text Encoder
- Model: `SakikoLab/Sakiko-Prompt-Gen-v1.0` from HuggingFace
- Hidden size: 1024
- Max token length: 384
- **System prompt support**: Yes (with chat template if available)
- Default system prompt: "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
- **Chat template**: Automatically detected and used if available in tokenizer
- No prompt weighting syntax support yet (planned for future)

### UNet
- Based on SDXL UNet architecture
- **Optimized for LSNet "See Large, Focus Small" design principles**
- Enhanced with LSNet components:
  - `LSConv`: Large-scale convolution with selective kernel attention
  - `RepVGGDW`: RepVGG depthwise convolution
  - `Conv2d_BN`: Convolution with batch normalization
- **Lightweight configuration optimized for efficiency**:
  - Channel multipliers: [1, 2, 4]
  - Attention resolutions: [4, 2]
  - Transformer depths: [0, 1, 3, 3] (7 blocks total - optimized for LSNet efficiency)
- Context dim: 1024 (matching text encoder hidden size)
- Context dim: 1024 (matching text encoder hidden size)

### VAE
- Flux VAE (reused)
- 16 latent channels
- Scale factor: 0.3611
- Shift factor: 0.1159
- 8x downsampling

### Sampling
- Flow matching / Rectified flow
- Euler sampling method
- Timestep schedule with adaptive shifting
- Classifier-free guidance support

## File Structure

```
sd-scripts/
├── library/
│   ├── sbxl_models.py           # Model definitions
│   ├── sbxl_utils.py            # Utility functions
│   ├── strategy_sbxl.py         # Training strategies
│   └── sbxl_train_util.py       # Training utilities
├── networks/
│   └── lora_sbxl.py             # LoRA implementation
├── sbxl_minimal_inference.py    # Inference script
├── sbxl_train_network.py        # LoRA training
├── sbxl_train.py                # Full model training
└── sbxl_train_control_net.py    # ControlNet training
```

## Usage

### 1. Inference

Basic inference with SBXL:

```bash
python sbxl_minimal_inference.py \
  --unet /path/to/sbxl_unet.safetensors \
  --vae /path/to/flux_vae.safetensors \
  --prompt "a beautiful landscape with mountains and lake" \
  --negative_prompt "blurry, low quality" \
  --system_prompt "You are an assistant designed to generate high-quality images..." \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --guidance_scale 3.5 \
  --seed 42 \
  --output output.png
```

Parameters:
- `--unet`: Path to SBXL UNet checkpoint
- `--vae`: Path to Flux VAE checkpoint
- `--text_encoder`: (Optional) Path to Sakiko text encoder, defaults to HuggingFace
- `--prompt`: Text prompt for generation
- `--negative_prompt`: Negative prompt for CFG
- `--system_prompt`: (Optional) Custom system prompt for text encoder
- `--width/height`: Image dimensions (default: 1024x1024)
- `--steps`: Number of sampling steps (default: 28)
- `--guidance_scale`: CFG scale (default: 3.5)
- `--seed`: Random seed
- `--dtype`: Model dtype (default: bf16)

### 2. LoRA Training

Train LoRA for SBXL:

```bash
python sbxl_train_network.py \
  --pretrained_model_name_or_path /path/to/sbxl_unet.safetensors \
  --vae /path/to/flux_vae.safetensors \
  --train_data_dir /path/to/training/data \
  --output_dir /path/to/output \
  --network_module networks.lora_sbxl \
  --network_dim 32 \
  --network_alpha 16 \
  --system_prompt "Custom system prompt for training" \
  --resolution 1024 \
  --train_batch_size 1 \
  --learning_rate 1e-4 \
  --max_train_epochs 10 \
  --save_every_n_epochs 1 \
  --mixed_precision bf16 \
  --cache_latents \
  --cache_text_encoder_outputs
```

Parameters:
- `--network_dim`: LoRA rank (default: 4, recommended: 16-64)
- `--network_alpha`: LoRA alpha for scaling (default: 1)
- `--system_prompt`: (Optional) Custom system prompt for text encoder
- `--cache_latents`: Cache latents to disk to save memory
- `--cache_text_encoder_outputs`: Cache text encoder outputs

### 3. Full Model Training

Comprehensive SBXL fine-tuning (UNet with optional text encoder updates):

```bash
python sbxl_train.py \
  --pretrained_model_name_or_path /path/to/sbxl_unet.safetensors \
  --vae /path/to/flux_vae.safetensors \
  --train_data_dir /path/to/training/data \
  --output_dir /path/to/output \
  --resolution 1024 \
  --train_batch_size 1 \
  --learning_rate 1e-5 \
  --max_train_epochs 10 \
  --cache_latents \
  --cache_text_encoder_outputs \
  --system_prompt "Custom prompt for training" \
  --mixed_precision bf16
```

The training loop mirrors Flux: flow-matching scheduler, SNR-aware velocity loss, gradient checkpointing, accelerator/deepspeed integration, configurable caching, and tracker logging are all supported out of the box.

### 4. ControlNet Training

Train an SBXL ControlNet initialised from the base UNet weights:

```bash
python sbxl_train_control_net.py \
  --pretrained_model_name_or_path /path/to/sbxl_unet.safetensors \
  --vae /path/to/flux_vae.safetensors \
  --train_data_dir /path/to/training/data \
  --conditioning_data_dir /path/to/conditioning/maps \
  --output_dir /path/to/output \
  --resolution 1024 \
  --conditioning_scale 1.0 \
  --cache_latents \
  --cache_text_encoder_outputs
```

The ControlNet trainer reuses SBXL-specific conditioning heads while exposing the same scheduler, caching, logging, and checkpoint capabilities as the Flux implementation.

## Training Dataset Format

Same as other models in this repo:

```
train_data_dir/
├── image1.png
├── image1.txt  (caption)
├── image2.png
├── image2.txt
└── ...
```

Or use dataset config files (`.toml` format).

## LSNet Integration

The UNet integrates LSNet components from the `lsnet/` folder:

- **LSConv**: Large-scale selective kernel convolution
  - Uses large kernel prediction (7x7)
  - Selective kernel attention
  - Replaces standard conv in most locations

- **RepVGGDW**: RepVGG-style depthwise convolution
  - Used in even-depth blocks
  - Efficient re-parameterization

- **Conv2d_BN**: Standard conv with batch norm
  - Used in patch embedding and transition layers

These components are automatically imported and used in `sbxl_models.py`.

## Model Components

### SBXLUNet2DConditionModel
Main UNet model with:
- Input/output channels: 4 (VAE latents)
- Model channels: 320
- Channel multipliers: [1, 2, 4]
- Attention heads: 64
- Context dimension: 1024
- Gradient checkpointing support

### LSNetResnetBlock2D
Enhanced ResNet block:
- Uses LSConv when input/output channels match
- Time embedding projection
- GroupNorm + SiLU activation

### Transformer2DModel
Standard transformer with:
- Self-attention
- Cross-attention (for text conditioning)
- Feed-forward network
- Layer normalization

## Flow Sampling

SBXL uses flow matching for sampling:

1. **Timestep Schedule**: Adaptive schedule based on image size
2. **Euler Method**: Simple and effective ODE solver
3. **CFG**: Classifier-free guidance with negative prompts
4. **Velocity Prediction**: Model predicts velocity field

## Attention Backends

Training scripts support multiple attention implementations for better throughput:

- `--use_flash_attn` enables PyTorch Flash Attention kernels (requires GPU with SDP support).
- `--mem_eff_attn` requests PyTorch's memory-efficient SDPA implementation.
- `--xformers` switches cross attention to xFormers' memory efficient attention (requires the `xformers` package).
- `--sdpa` keeps the standard PyTorch SDPA path (default).

## Requirements

- PyTorch >= 2.0
- transformers
- accelerate
- safetensors
- einops
- PIL
- numpy

LSNet components require:
- timm
- torch

## Text Encoder Features

### System Prompt

SBXL supports system prompts for the text encoder, similar to the implementation in `text_encoder_example.py`. The system prompt helps guide the text encoder to generate better image-text alignment.

**Default System Prompt**:
```
You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts.
```

**Custom System Prompt**:
```bash
# Inference
python sbxl_minimal_inference.py \
  --prompt "a cat" \
  --system_prompt "You are a specialized art generator..." \
  ...

# Training
python sbxl_train_network.py \
  --system_prompt "Custom training prompt..." \
  ...
```

### Chat Template Support

If the Sakiko tokenizer has a `chat_template` defined, SBXL will automatically use it:

```python
# With chat template
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
full_prompt = tokenizer.apply_chat_template(messages, ...)

# Without chat template (fallback)
full_prompt = f'{system_prompt} <Prompt Start> {user_prompt}'
```

The system automatically detects and uses the appropriate method.

### Negative Prompts

Negative prompts **do not** use the system prompt to avoid contamination:

```python
# Positive prompt: uses system prompt
positive = encode_prompts(prompt, system_prompt=system_prompt)

# Negative prompt: no system prompt
negative = encode_prompts(negative_prompt, system_prompt="")
```

## Known Limitations

1. **Prompt weighting syntax not supported** – Token weights are uniform until weighting-aware tokenization is added
2. **No public SBXL checkpoints yet** – Train from scratch or adapt weights from related architectures
3. **Single sampler implementation** – Only the Euler rectified-flow sampler is bundled today

## Future Work

- [ ] Implement prompt weighting syntax and caching support
- [ ] Add additional samplers (DPM-Solver++, Heun, etc.)
- [ ] Broaden LSNet component coverage and fused kernels
- [ ] Multi-resolution / multi-aspect training recipes
- [ ] Knowledge distillation guidance and evaluation harnesses

## Credits

- LSNet: From `lsnet/` folder in this repository
- Flux: VAE and sampling inspiration
- SDXL: UNet architecture base
- Lumina: Text encoder integration patterns
- Sakiko: Text encoder model (SakikoLab/Sakiko-Prompt-Gen-v1.0)

## License

Follow the same license as the parent sd-scripts repository.

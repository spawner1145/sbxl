# SBXL Usage Examples

This document provides practical examples for using SBXL with various configurations.

## Basic Inference

### Simple Generation

```bash
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --prompt "a serene mountain landscape at sunset" \
  --output output.png
```

### With Negative Prompt and CFG

```bash
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --prompt "a beautiful anime girl, detailed face, high quality" \
  --negative_prompt "blurry, low quality, distorted, ugly" \
  --guidance_scale 7.5 \
  --steps 50 \
  --seed 42 \
  --output anime_girl.png
```

### Custom System Prompt

```bash
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --prompt "a photorealistic portrait" \
  --system_prompt "You are a professional photographer AI specialized in portrait photography." \
  --output portrait.png
```

### Batch Generation

```bash
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --prompt "abstract art, colorful" \
  --batch_size 4 \
  --seed 1000 \
  --output_dir outputs/abstract/
```

### High Resolution

```bash
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --prompt "detailed cityscape" \
  --width 2048 \
  --height 1536 \
  --steps 40 \
  --output cityscape_4k.png
```

## LoRA Training

### Basic LoRA Training

```bash
python sbxl_train_network.py \
  --pretrained_model_name_or_path models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --train_data_dir data/my_dataset/ \
  --output_dir outputs/lora_training/ \
  --output_name my_style_lora \
  --network_module networks.lora_sbxl \
  --network_dim 32 \
  --network_alpha 16 \
  --resolution 1024 \
  --train_batch_size 1 \
  --learning_rate 1e-4 \
  --max_train_epochs 10 \
  --save_every_n_epochs 2 \
  --mixed_precision bf16 \
  --cache_latents
```

### Advanced LoRA Training with Custom System Prompt

```bash
python sbxl_train_network.py \
  --pretrained_model_name_or_path models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --train_data_dir data/anime_style/ \
  --output_dir outputs/anime_lora/ \
  --output_name anime_style_lora \
  --system_prompt "You are an anime art generator specialized in creating high-quality anime illustrations." \
  --network_module networks.lora_sbxl \
  --network_dim 64 \
  --network_alpha 32 \
  --network_dropout 0.1 \
  --resolution 1024 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --max_train_epochs 20 \
  --save_every_n_epochs 5 \
  --mixed_precision bf16 \
  --cache_latents \
  --cache_text_encoder_outputs \
  --caption_extension .txt \
  --shuffle_caption \
  --keep_tokens 1 \
  --max_token_length 384 \
  --xformers \
  --gradient_checkpointing
```

> Tip: Add `--use_flash_attn` if your environment supports PyTorch Flash Attention kernels, or `--mem_eff_attn` to prefer memory-efficient SDPA.

### Character LoRA Training

```bash
python sbxl_train_network.py \
  --pretrained_model_name_or_path models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --train_data_dir data/character/ \
  --reg_data_dir data/reg_images/ \
  --output_dir outputs/character_lora/ \
  --output_name character_v1 \
  --network_dim 128 \
  --network_alpha 64 \
  --resolution "512,768,1024" \
  --enable_bucket \
  --min_bucket_reso 512 \
  --max_bucket_reso 2048 \
  --train_batch_size 4 \
  --learning_rate 1e-4 \
  --max_train_steps 5000 \
  --save_every_n_steps 500 \
  --mixed_precision bf16
```

### Text Encoder LoRA Training

```bash
python sbxl_train_network.py \
  --pretrained_model_name_or_path models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --text_encoder_path SakikoLab/Sakiko-Prompt-Gen-v1.0 \
  --train_data_dir data/concept/ \
  --output_dir outputs/concept_lora/ \
  --network_train_text_encoder_only \
  --network_dim 32 \
  --network_alpha 16 \
  --learning_rate 5e-5 \
  --max_train_epochs 50
```

## Dataset Preparation

### Directory Structure

```
data/
├── train/
│   ├── image1.png
│   ├── image1.txt
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
└── reg/  (optional, for regularization)
    ├── reg1.png
    ├── reg1.txt
    └── ...
```

### Caption Files

Each image should have a corresponding `.txt` file with the same base name:

**image1.txt**:
```
a beautiful landscape with mountains and lake, sunset lighting, high quality photograph
```

### Using Dataset Config (TOML)

Create a config file `dataset.toml`:

```toml
[general]
resolution = 1024
shuffle_caption = true
keep_tokens = 1
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = 4

  [[datasets.subsets]]
  image_dir = "data/train"
  num_repeats = 10
  
  [[datasets.subsets]]
  image_dir = "data/reg"
  num_repeats = 1
  is_reg = true
```

Then use it:

```bash
python sbxl_train_network.py \
  --dataset_config dataset.toml \
  --pretrained_model_name_or_path models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --output_dir outputs/
```

## Advanced Inference Techniques

### Deterministic Generation

```bash
# Same seed = same image
python sbxl_minimal_inference.py \
  --prompt "test prompt" \
  --seed 12345 \
  --output test1.png

python sbxl_minimal_inference.py \
  --prompt "test prompt" \
  --seed 12345 \
  --output test2.png
# test1.png and test2.png will be identical
```

### Aspect Ratio Variations

```bash
# Portrait
python sbxl_minimal_inference.py \
  --width 768 --height 1344 \
  --prompt "portrait of a person"

# Landscape
python sbxl_minimal_inference.py \
  --width 1344 --height 768 \
  --prompt "wide landscape view"

# Square
python sbxl_minimal_inference.py \
  --width 1024 --height 1024 \
  --prompt "centered composition"
```

### Different Sampling Settings

```bash
# Fast (fewer steps)
python sbxl_minimal_inference.py \
  --steps 20 \
  --prompt "quick test"

# Quality (more steps)
python sbxl_minimal_inference.py \
  --steps 50 \
  --prompt "high quality render"

# Strong guidance
python sbxl_minimal_inference.py \
  --guidance_scale 10.0 \
  --prompt "very specific requirements"

# Weak guidance (more creative)
python sbxl_minimal_inference.py \
  --guidance_scale 2.0 \
  --prompt "creative interpretation"
```

## System Prompt Examples

### For Photorealistic Images

```bash
--system_prompt "You are a professional photographer AI. Generate photorealistic images with accurate lighting, composition, and detail."
```

### For Artistic Styles

```bash
--system_prompt "You are an art director specialized in creating artistic illustrations with strong visual impact and aesthetic appeal."
```

### For Anime/Manga

```bash
--system_prompt "You are an anime illustration AI specialized in creating high-quality anime and manga style artwork with proper proportions and expressive characters."
```

### For Technical/Architectural

```bash
--system_prompt "You are a technical visualization AI specialized in creating accurate architectural and engineering visualizations."
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--train_batch_size 1

# Enable gradient checkpointing
--gradient_checkpointing

# Cache latents to disk
--cache_latents
--cache_text_encoder_outputs

# Use lower precision
--mixed_precision bf16
```

### Slow Training

```bash
# Enable xformers
--xformers

# Cache everything
--cache_latents
--cache_text_encoder_outputs_to_disk

# Use larger batch size with gradient accumulation
--train_batch_size 4
--gradient_accumulation_steps 4
```

### Poor Image Quality

```bash
# Increase steps
--steps 50

# Adjust guidance scale
--guidance_scale 7.5

# Use better prompt
--system_prompt "..."
--prompt "detailed, high quality, ..."

# Try different seeds
--seed 42
```

## Integration with Other Tools

### Using with LoRA

```bash
# After training a LoRA
python sbxl_minimal_inference.py \
  --unet models/sbxl_unet.safetensors \
  --vae models/flux_vae.safetensors \
  --lora_weights outputs/my_lora.safetensors;0.8 \
  --prompt "test with lora"
```

Multiple LoRAs can be stacked by repeating `--lora_weights path;multiplier`. Use `--merge_lora_weights` to bake them into the base UNet before generation.

## Performance Tips

1. **Use caching**: `--cache_latents` and `--cache_text_encoder_outputs` for training
2. **Enable xformers**: `--xformers` for faster attention
3. **Mixed precision**: `--mixed_precision bf16` for memory efficiency
4. **Gradient checkpointing**: `--gradient_checkpointing` for large models
5. **Batch size tuning**: Find optimal balance between speed and memory
6. **Resolution bucketing**: `--enable_bucket` for mixed resolution training

## Best Practices

1. **Start with defaults**: Use default parameters first, then tune
2. **Test system prompts**: Experiment with different system prompts for your use case
3. **Caption quality**: Good captions = good results
4. **Learning rate**: Start with 1e-4 for LoRA, adjust based on results
5. **Regular checkpoints**: Save frequently to avoid losing progress
6. **Validation**: Generate test images during training to monitor progress

import torch
import numpy as np
from PIL import Image
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from library import sbxl_utils

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    if img1.dim() == 4:
        img1 = img1.squeeze(0)
    if img2.dim() == 4:
        img2 = img2.squeeze(0)
    img1_gray = 0.2989 * img1[0] + 0.5870 * img1[1] + 0.1140 * img1[2]
    img2_gray = 0.2989 * img2[0] + 0.5870 * img2[1] + 0.1140 * img2[2]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    kernel = kernel.to(img1.device)
    mu1 = torch.nn.functional.conv2d(img1_gray.unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)
    mu2 = torch.nn.functional.conv2d(img2_gray.unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d((img1_gray.unsqueeze(0).unsqueeze(0) - mu1) ** 2, kernel, padding=window_size//2)
    sigma2_sq = torch.nn.functional.conv2d((img2_gray.unsqueeze(0).unsqueeze(0) - mu2) ** 2, kernel, padding=window_size//2)
    sigma12 = torch.nn.functional.conv2d((img1_gray.unsqueeze(0).unsqueeze(0) - mu1) * (img2_gray.unsqueeze(0).unsqueeze(0) - mu2), kernel, padding=window_size//2)
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return float(torch.mean(ssim_map).item())

def process_image_with_vae_standard(image_path: str, vae, device: torch.device) -> Dict[str, Any]:
    try:
        test_image_pil = Image.open(image_path).convert('RGB')
        original_size = test_image_pil.size
        new_size = (original_size[0] // 16 * 16, original_size[1] // 16 * 16)

        if new_size != original_size:
            test_image_pil = test_image_pil.resize(new_size, Image.LANCZOS)
            print(f"    Adjusted size from {original_size} to {new_size} (VAE requirement)")
            original_size = new_size
        print(f"    Image size: {original_size}, processing...")
        test_image_np = np.array(test_image_pil).astype(np.float32) / 127.5 - 1.0
        test_image = torch.from_numpy(test_image_np).permute(2, 0, 1).unsqueeze(0)
        print("    Encoding...")
        with torch.no_grad():
            latents = vae.encode(test_image.to(vae.device).to(vae.dtype))
        print("    Decoding...")
        with torch.no_grad():
            reconstructed = vae.decode(latents)
        mse = torch.mean((test_image - reconstructed.cpu()) ** 2).item()
        psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))) if mse > 0 else float('inf')
        ssim_value = compute_ssim(test_image, reconstructed.cpu())

        print(f"    MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")

        return {
            'original_size': original_size,
            'latent_shape': list(latents.shape),
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim_value),
            'test_image': test_image,
            'reconstructed': reconstructed,
            'latents': latents
        }

    except Exception as e:
        print(f"    Error in process_image_with_vae_standard: {e}")
        raise


def process_image_with_vae_tiled(image_path: str, vae, device: torch.device, tile_size: int = 1024, overlap: int = 64) -> Dict[str, Any]:
    try:
        test_image_pil = Image.open(image_path).convert('RGB')
        original_size = test_image_pil.size
        width, height = original_size
        print(f"    Image size: {original_size}, using tiled processing (tile_size={tile_size}, overlap={overlap})")
        test_image_np = np.array(test_image_pil).astype(np.float32) / 127.5 - 1.0
        test_image_full = torch.from_numpy(test_image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        tile_size = (tile_size // 16) * 16
        stride = tile_size - overlap
        num_tiles_x = int(np.ceil((width - overlap) / stride))
        num_tiles_y = int(np.ceil((height - overlap) / stride))
        print(f"    Tiles: {num_tiles_x}x{num_tiles_y}")
        reconstructed_full = torch.zeros_like(test_image_full)
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                y_start = i * stride
                x_start = j * stride
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                if y_end - y_start < tile_size and y_start > 0:
                    y_start = max(0, y_end - tile_size)
                if x_end - x_start < tile_size and x_start > 0:
                    x_start = max(0, x_end - tile_size)
                tile = test_image_full[:, :, y_start:y_end, x_start:x_end]
                if tile.shape[2] < 64 or tile.shape[3] < 64:
                    continue
                tile_h, tile_w = tile.shape[2], tile.shape[3]
                tile_h = (tile_h // 16) * 16
                tile_w = (tile_w // 16) * 16
                tile = tile[:, :, :tile_h, :tile_w]
                print(f"      Processing tile ({j},{i}): {tile.shape}")

                with torch.no_grad():
                    latents = vae.encode(tile.to(vae.device).to(vae.dtype))

                with torch.no_grad():
                    reconstructed_tile = vae.decode(latents).cpu()

                reconstructed_full[:, :, y_start:y_start+tile_h, x_start:x_start+tile_w] = reconstructed_tile

        mse = torch.mean((test_image_full - reconstructed_full) ** 2).item()
        psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))) if mse > 0 else float('inf')
        ssim_value = compute_ssim(test_image_full, reconstructed_full)

        print(f"    MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")

        return {
            'original_size': original_size,
            'latent_shape': None,
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim_value),
            'test_image': test_image_full,
            'reconstructed': reconstructed_full,
            'latents': None
        }

    except Exception as e:
        print(f"    Error in process_image_with_vae_tiled: {e}")
        raise

def save_reconstruction_results(image_path: str, reconstructed: torch.Tensor, test_image: torch.Tensor,
                               original_size: tuple, output_dir: Path, vae_name: str):
    stem = Path(image_path).stem
    suffix = Path(image_path).suffix

    reconstructed_np = ((reconstructed[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    reconstructed_img = Image.fromarray(reconstructed_np)
    if reconstructed_img.size != original_size:
        reconstructed_img = reconstructed_img.resize(original_size, Image.LANCZOS)

    recons_filename = f"{stem}-{vae_name}-recons{suffix}"
    reconstructed_img.save(output_dir / recons_filename)

    diff = torch.abs(reconstructed.cpu() - test_image)
    diff_np = (diff[0].permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
    diff_img = Image.fromarray(diff_np)
    if diff_img.size != original_size:
        diff_img = diff_img.resize(original_size, Image.LANCZOS)

    diff_filename = f"{stem}-{vae_name}-diff{suffix}"
    diff_img.save(output_dir / diff_filename)

def batch_process_images(image_dir: str, vae_paths: List[str], max_image_size: int = 2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = Path(image_dir)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Processing with {len(vae_paths)} VAEs: {vae_paths}")

    all_stats = {}

    for vae_path in vae_paths:
        vae_name = Path(vae_path).stem
        print(f"\nProcessing with VAE: {vae_name}")

        output_dir = image_dir / vae_name
        output_dir.mkdir(exist_ok=True)

        vae = sbxl_utils.load_vae(vae_path, torch.float16, device)
        vae.to(device)
        vae.eval()

        print(f"VAE loaded: device={vae.device}, dtype={vae.dtype}")

        image_stats = {}

        for image_path in image_files:
            print(f"  Processing: {image_path.name}")

            try:
                test_image_pil_temp = Image.open(str(image_path))
                max_dim = max(test_image_pil_temp.size)

                if max_dim > max_image_size:
                    print(f"    Using tiled processing for large image (max_dim={max_dim})")
                    result = process_image_with_vae_tiled(str(image_path), vae, device)
                else:
                    print(f"    Using standard processing (max_dim={max_dim})")
                    result = process_image_with_vae_standard(str(image_path), vae, device)

                original_size = result['original_size']
                test_image = result['test_image']
                reconstructed = result['reconstructed']
                save_reconstruction_results(str(image_path), reconstructed, test_image,
                                          original_size, output_dir, vae_name)
                image_stats[image_path.name] = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in result.items()
                    if k not in ['test_image', 'reconstructed', 'latents']
                }

                del result, test_image, reconstructed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error processing {image_path.name}: {e}")
                image_stats[image_path.name] = {'error': str(e)}

        valid_stats = [stats for stats in image_stats.values() if 'error' not in stats]

        vae_stats_data = {
            'vae_path': vae_path,
            'vae_name': vae_name,
            'processed_at': str(Path(__file__).parent),
            'images': image_stats
        }

        with open(output_dir / 'reconstruction_stats.json', 'w', encoding='utf-8') as f:
            json.dump(vae_stats_data, f, indent=2, ensure_ascii=False)

        vae_summary = {
            'vae_name': vae_name,
            'num_images': len(valid_stats),
        }

        if valid_stats:
            vae_summary.update({
                'avg_mse': float(np.mean([s['mse'] for s in valid_stats])),
                'avg_psnr': float(np.mean([s['psnr'] for s in valid_stats])),
                'min_psnr': float(min([s['psnr'] for s in valid_stats])),
                'max_psnr': float(max([s['psnr'] for s in valid_stats])),
                'std_psnr': float(np.std([s['psnr'] for s in valid_stats])),
            })

            if any(s.get('ssim') is not None for s in valid_stats):
                ssim_values = [s['ssim'] for s in valid_stats if s.get('ssim') is not None]
                vae_summary.update({
                    'avg_ssim': float(np.mean(ssim_values)),
                    'std_ssim': float(np.std(ssim_values)),
                })

        all_stats[vae_name] = vae_summary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nBatch processing completed. Results saved to {image_dir}")

    overall_summary = {
        'total_images': len(image_files),
        'vaes_processed': list(all_stats.keys()),
        'vae_summaries': all_stats
    }

    with open(image_dir / 'overall_summary.json', 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    image_dir = "duskming"
    vae_paths = [
        "ae.safetensors",
        "MS-LC-EQ-D-VR VAE FLUX.safetensors",
        "Pad Flux EQ v2 B1.safetensors",
    ]
    max_image_size = 2048  # 分块处理阈值

    print("Starting batch VAE reconstruction test...")
    print(f"Image directory: {image_dir}")
    print(f"VAE paths: {vae_paths}")
    print(f"Max image size: {max_image_size}")
    print()

    batch_process_images(image_dir, vae_paths, max_image_size)
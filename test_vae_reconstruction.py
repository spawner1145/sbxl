import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from library import sbxl_utils

def test_vae_reconstruction(image_path="image.png"):
    vae_path = "ae.safetensors"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = sbxl_utils.load_vae(vae_path, torch.float16, device)
    vae.to(device)
    vae.eval()

    print(f"VAE loaded successfully. Scale factor: {vae.scale_factor}, Shift factor: {vae.shift_factor}")
    print(f"VAE device: {vae.device}, dtype: {vae.dtype}")

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return None

    test_image_pil = Image.open(image_path).convert('RGB')
    original_size = test_image_pil.size
    print(f"Loaded image: {original_size}")
    test_image_np = np.array(test_image_pil).astype(np.float32) / 127.5 - 1.0
    test_image = torch.from_numpy(test_image_np).permute(2, 0, 1).unsqueeze(0)
    print(f"Test image shape: {test_image.shape}, range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    with torch.no_grad():
        latents = vae.encode(test_image.to(vae.device).to(vae.dtype))
        print(f"Latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")

    with torch.no_grad():
        reconstructed = vae.decode(latents)
        print(f"Reconstructed shape: {reconstructed.shape}, range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    mse = torch.mean((test_image - reconstructed.cpu()) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    reconstructed_np = ((reconstructed[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    reconstructed_img = Image.fromarray(reconstructed_np)
    if reconstructed_img.size != original_size:
        reconstructed_img = reconstructed_img.resize(original_size, Image.LANCZOS)
    reconstructed_img.save("test_vae_reconstructed.png")

    diff = torch.abs(reconstructed.cpu() - test_image)
    diff_np = (diff[0].permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
    diff_img = Image.fromarray(diff_np)
    if diff_img.size != original_size:
        diff_img = diff_img.resize(original_size, Image.LANCZOS)
    diff_img.save("test_vae_difference.png")

    print("Images saved as test_vae_reconstructed.png and test_vae_difference.png")

    return psnr.item()

if __name__ == "__main__":
    input_image_path = "test_vae_original.png"
    
    psnr = test_vae_reconstruction(input_image_path)
    print(f"VAE reconstruction test completed. PSNR: {psnr:.2f} dB")
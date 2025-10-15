#!/usr/bin/env python3
"""
Create dummy test images for SBXL training
"""

import os
import numpy as np
from PIL import Image

def create_dummy_images():
    """Create dummy RGB images for testing"""
    os.makedirs("test_images", exist_ok=True)

    # Create 4 dummy images of different sizes
    sizes = [(512, 512), (768, 512), (512, 768), (1024, 1024)]

    for i, (width, height) in enumerate(sizes):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')

        # Save image
        img_path = f"test_images/test_image_{i+1:02d}.png"
        img.save(img_path)
        print(f"Created {img_path} ({width}x{height})")

        # Create corresponding caption file
        caption_path = f"test_images/test_image_{i+1:02d}.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(f"test image {i+1}, anime style, high quality")
        print(f"Created {caption_path}")

if __name__ == "__main__":
    create_dummy_images()
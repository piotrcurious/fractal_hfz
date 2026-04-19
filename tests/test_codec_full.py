import numpy as np
import os
from PIL import Image
from fractal_hfz_codec import FractalHybridCodec

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def test_all_images():
    codec = FractalHybridCodec()
    image_dir = 'tests/images'

    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found")
        return

    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    images.sort()

    print(f"{'Image':<20} | {'PSNR':<10} | {'BPP':<10}")
    print("-" * 46)

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = np.array(Image.open(img_path))

        # Compress
        blob = codec.compress(img, quality=25)

        # Decompress
        recon = codec.decompress(blob)

        # Metrics
        psnr = calculate_psnr(img, recon)
        bpp = (len(blob) * 8) / (img.shape[0] * img.shape[1])

        print(f"{img_name:<20} | {psnr:<10.2f} | {bpp:<10.3f}")

        # Basic sanity check
        assert psnr > 20, f"PSNR for {img_name} is too low: {psnr}"

if __name__ == "__main__":
    test_all_images()

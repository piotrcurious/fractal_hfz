import os
import subprocess
import time
import numpy as np
from PIL import Image
from pathlib import Path
import json

def calculate_psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB')).astype(np.float64)
    img2 = np.array(Image.open(img2_path).convert('RGB')).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def run_benchmark(codec_path, input_dir, output_dir, quality_levels=[25, 55, 85]):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_dir.glob("*.png")))
    results = []

    for img_path in images:
        for q in quality_levels:
            stem = img_path.stem
            compressed_path = output_dir / f"{stem}_q{q}.hfz"
            reconstructed_path = output_dir / f"{stem}_q{q}_recon.png"

            # Encode
            start_time = time.time()
            subprocess.run([
                "python3", str(codec_path), "encode",
                str(img_path), str(compressed_path),
                "--quality", str(q)
            ], check=True, capture_output=True)
            encode_time = time.time() - start_time

            # Decode
            start_time = time.time()
            subprocess.run([
                "python3", str(codec_path), "decode",
                str(compressed_path), str(reconstructed_path)
            ], check=True, capture_output=True)
            decode_time = time.time() - start_time

            # Metrics
            file_size = compressed_path.stat().st_size
            psnr = calculate_psnr(img_path, reconstructed_path)

            res = {
                "image": stem,
                "quality": q,
                "size": file_size,
                "psnr": psnr,
                "encode_time": encode_time,
                "decode_time": decode_time
            }
            results.append(res)
            print(f"Image: {stem}, Q: {q}, Size: {file_size}, PSNR: {psnr:.2f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python3 evaluate_codec.py <codec_path> <input_dir> <output_dir>")
        sys.exit(1)

    codec = sys.argv[1]
    input_d = sys.argv[2]
    output_d = sys.argv[3]

    run_benchmark(codec, input_d, output_d)

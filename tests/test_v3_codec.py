import os
import numpy as np
from PIL import Image
from v3.fractal_hfz_codec_v3 import FractalHybridCodec

def test_v3_grayscale_roundtrip():
    data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    codec = FractalHybridCodec(block_size=8)
    blob = codec.compress(data, quality=85)
    recon = codec.decompress(blob)
    assert recon.shape == data.shape
    # For random noise, PSNR won't be great, but it should decode.
    assert recon.dtype == np.uint8

def test_v3_rgb_roundtrip():
    data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    codec = FractalHybridCodec(block_size=8)
    blob = codec.compress(data, quality=85)
    recon = codec.decompress(blob)
    assert recon.shape == data.shape
    assert recon.dtype == np.uint8

def test_v3_camera_roundtrip():
    img = Image.open("tests/images/camera.png").convert("L")
    data = np.array(img)
    codec = FractalHybridCodec(block_size=8)
    blob = codec.compress(data, quality=55)
    recon = codec.decompress(blob)
    assert recon.shape == data.shape
    from skimage.metrics import peak_signal_noise_ratio as psnr
    p = psnr(data, recon)
    assert p > 35

def test_v3_non_default_params():
    img = Image.open("tests/images/camera.png").convert("L")
    data = np.array(img)
    # Use non-default block_size and residual_levels
    codec = FractalHybridCodec(block_size=16, residual_levels=3)
    blob = codec.compress(data, quality=55)

    # Decoder should handle it
    decoder = FractalHybridCodec(block_size=8, residual_levels=2) # default
    recon = decoder.decompress(blob)

    assert recon.shape == data.shape
    assert decoder.block_size == 16
    assert decoder.residual_levels == 3

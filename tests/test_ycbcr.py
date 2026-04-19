
import numpy as np
from fractal_hfz_codec import rgb_to_ycbcr, ycbcr_to_rgb
from skimage import data

def test_ycbcr_roundtrip():
    img = data.astronaut()
    # Test without subsampling
    channels = rgb_to_ycbcr(img, subsample=False)
    reconstructed = ycbcr_to_rgb(channels[0], channels[1], channels[2])

    diff = np.abs(img.astype(np.float64) - reconstructed.astype(np.float64))
    print(f"No subsample - Max diff: {np.max(diff)}")

    if np.max(diff) > 2:
        print("FAILED: Roundtrip (no subsample) difference too large")
    else:
        print("PASSED: YCbCr roundtrip (no subsample) successful")

if __name__ == "__main__":
    test_ycbcr_roundtrip()

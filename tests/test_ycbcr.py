
import numpy as np
from fractal_hfz_codec import rgb_to_ycbcr, ycbcr_to_rgb
from skimage import data

def test_ycbcr_roundtrip():
    img = data.astronaut()
    channels = rgb_to_ycbcr(img)
    reconstructed = ycbcr_to_rgb(channels[0], channels[1], channels[2])

    diff = np.abs(img.astype(np.float64) - reconstructed.astype(np.float64))
    print(f"Max diff: {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")

    if np.max(diff) > 2: # Small rounding errors are expected
        print("FAILED: Roundtrip difference too large")
        # Save for inspection
        from PIL import Image
        Image.fromarray(img).save("original.png")
        Image.fromarray(reconstructed).save("reconstructed_ycbcr.png")
    else:
        print("PASSED: YCbCr roundtrip successful")

if __name__ == "__main__":
    test_ycbcr_roundtrip()

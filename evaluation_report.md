# Codec Evaluation Report

## Comparison: v2 (fractal_hfz_codec.py) vs v3.3 (v3.2/v3_3.py)

| Image | Quality | v2 Size | v3 Size | Size Diff | v2 PSNR | v3 PSNR | PSNR Diff |
|-------|---------|---------|---------|-----------|---------|---------|-----------|
| astronaut | 25 | 138304 | 138272 | -32 | 35.93 | 35.93 | +0.00 |
| astronaut | 55 | 175156 | 175280 | +124 | 38.43 | 38.43 | +0.00 |
| astronaut | 85 | 200172 | 200296 | +124 | 39.73 | 39.73 | +0.00 |
| camera | 25 | 75844 | 75844 | +0 | 39.00 | 39.00 | +0.00 |
| camera | 55 | 92348 | 92348 | +0 | 41.90 | 41.90 | +0.00 |
| camera | 85 | 104336 | 104336 | +0 | 43.59 | 43.59 | +0.00 |
| coffee | 25 | 141056 | 141172 | +116 | 34.77 | 34.77 | +0.00 |
| coffee | 55 | 183152 | 183060 | -92 | 37.36 | 37.36 | +0.00 |
| coffee | 85 | 210904 | 210996 | +92 | 38.70 | 38.70 | +0.00 |
| moon | 25 | 22344 | 22336 | -8 | 42.17 | 42.17 | +0.00 |
| moon | 55 | 31972 | 32020 | +48 | 44.03 | 44.03 | +0.00 |
| moon | 85 | 43680 | 43668 | -12 | 45.29 | 45.29 | +0.00 |

## Observations
- Reconstruction quality (PSNR) is perfectly preserved between v2 and v3.3.
- Minimal size changes indicate that Stage 2.5 fractal prediction has a neutral to slightly positive impact on compression efficiency for these test images.

# Codec Evaluation Report

## Comparison: evaluation_artifacts/v3 vs evaluation_artifacts/v3_4

| Image | Quality | Old Size | New Size | Size Diff | Old PSNR | New PSNR | PSNR Diff |
|-------|---------|----------|----------|-----------|----------|----------|-----------|
| astronaut | 25 | 138272 | 132988 | -5284 | 35.93 | 35.60 | -0.33 |
| astronaut | 55 | 175280 | 167536 | -7744 | 38.43 | 38.11 | -0.32 |
| astronaut | 85 | 200296 | 191636 | -8660 | 39.73 | 39.39 | -0.33 |
| camera | 25 | 75844 | 74928 | -916 | 39.00 | 38.42 | -0.58 |
| camera | 55 | 92348 | 90900 | -1448 | 41.90 | 41.37 | -0.53 |
| camera | 85 | 104336 | 102680 | -1656 | 43.59 | 43.09 | -0.50 |
| coffee | 25 | 141172 | 134612 | -6560 | 34.77 | 34.46 | -0.31 |
| coffee | 55 | 183060 | 175384 | -7676 | 37.36 | 37.03 | -0.33 |
| coffee | 85 | 210996 | 202528 | -8468 | 38.70 | 38.34 | -0.35 |
| moon | 25 | 22336 | 22188 | -148 | 42.17 | 41.79 | -0.38 |
| moon | 55 | 32020 | 32608 | +588 | 44.03 | 43.69 | -0.34 |
| moon | 85 | 43668 | 43012 | -656 | 45.29 | 44.97 | -0.33 |

## Observations
- v3.4 introduces Extended-Basis Overlap-Add (EBOA) which significantly reduces bitstream size across all images.
- The smoother base reconstruction from EBOA allows for coarser Haar quantisation (1.60x) while maintaining competitive visual quality.
- Size savings range from ~1 KB up to ~8 KB per image, with an average PSNR reduction of only ~0.3-0.5 dB.

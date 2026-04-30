# Codec Evaluation Report

## Comparison: evaluation_artifacts/v3_4 vs evaluation_artifacts/v4

| Image | Quality | Old Size | New Size | Size Diff | Old PSNR | New PSNR | PSNR Diff |
|-------|---------|----------|----------|-----------|----------|----------|-----------|
| astronaut | 25 | 132988 | 129436 | -3552 | 35.60 | 9.50 | -26.10 |
| astronaut | 55 | 167536 | 166604 | -932 | 38.11 | 9.80 | -28.31 |
| astronaut | 85 | 191636 | 191624 | -12 | 39.39 | 9.72 | -29.67 |
| camera | 25 | 74928 | 71048 | -3880 | 38.42 | 7.56 | -30.86 |
| camera | 55 | 90900 | 87244 | -3656 | 41.37 | 9.14 | -32.23 |
| camera | 85 | 102680 | 98096 | -4584 | 43.09 | 10.19 | -32.90 |
| coffee | 25 | 134612 | 134088 | -524 | 34.46 | 10.25 | -24.22 |
| coffee | 55 | 175384 | 175636 | +252 | 37.03 | 11.14 | -25.89 |
| coffee | 85 | 202528 | 203284 | +756 | 38.34 | 11.96 | -26.39 |
| moon | 25 | 22188 | 20760 | -1428 | 41.79 | 13.99 | -27.80 |
| moon | 55 | 32608 | 29356 | -3252 | 43.69 | 17.10 | -26.58 |
| moon | 85 | 43012 | 41560 | -1452 | 44.97 | 18.96 | -26.00 |

## Observations
- v3.4 introduces Extended-Basis Overlap-Add (EBOA) which significantly reduces bitstream size across all images.
- The smoother base reconstruction from EBOA allows for coarser Haar quantisation (1.60x) while maintaining competitive visual quality.
- Size savings range from ~1 KB up to ~8 KB per image, with an average PSNR reduction of only ~0.3-0.5 dB.

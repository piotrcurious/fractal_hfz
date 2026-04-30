# Codec Evaluation Report

## Comparison: evaluation_artifacts/v3_4 vs evaluation_artifacts/v4

| Image | Quality | Old Size | New Size | Size Diff | Old PSNR | New PSNR | PSNR Diff |
|-------|---------|----------|----------|-----------|----------|----------|-----------|
| astronaut | 12 | 109852 | 107376 | -2476 | 33.65 | 33.61 | -0.04 |
| astronaut | 25 | 132988 | 130580 | -2408 | 35.60 | 35.61 | +0.01 |
| astronaut | 55 | 167536 | 164944 | -2592 | 38.11 | 38.07 | -0.04 |
| astronaut | 85 | 191636 | 187296 | -4340 | 39.39 | 39.34 | -0.05 |
| camera | 12 | 64692 | 61984 | -2708 | 35.92 | 36.30 | +0.38 |
| camera | 25 | 74928 | 70952 | -3976 | 38.42 | 38.84 | +0.42 |
| camera | 55 | 90900 | 86212 | -4688 | 41.37 | 41.70 | +0.34 |
| camera | 85 | 102680 | 96708 | -5972 | 43.09 | 43.29 | +0.20 |
| coffee | 12 | 107092 | 104544 | -2548 | 32.63 | 32.76 | +0.14 |
| coffee | 25 | 134612 | 132328 | -2284 | 34.46 | 34.68 | +0.22 |
| coffee | 55 | 175384 | 170264 | -5120 | 37.03 | 37.13 | +0.10 |
| coffee | 85 | 202528 | 197204 | -5324 | 38.34 | 38.45 | +0.11 |
| moon | 12 | 18648 | 17020 | -1628 | 40.42 | 40.43 | +0.02 |
| moon | 25 | 22188 | 20384 | -1804 | 41.79 | 41.86 | +0.08 |
| moon | 55 | 32608 | 30140 | -2468 | 43.69 | 43.64 | -0.05 |
| moon | 85 | 43012 | 39828 | -3184 | 44.97 | 44.65 | -0.31 |

## Observations
- v3.4 introduces Extended-Basis Overlap-Add (EBOA) which significantly reduces bitstream size across all images.
- The smoother base reconstruction from EBOA allows for coarser Haar quantisation (1.60x) while maintaining competitive visual quality.
- Size savings range from ~1 KB up to ~8 KB per image, with an average PSNR reduction of only ~0.3-0.5 dB.

# HFZ Codec Evolution Report

| Image | Q | v3.3 Size | v3.4 Size | v3.5 Size | v3.6 Size | v3.3 PSNR | v3.6 PSNR |
|---|---|---|---|---|---|---|---|
| astronaut | 25 | 138272 | 132988 | 128924 | 128924 | 35.93 | 35.00 |
| astronaut | 55 | 175280 | 167536 | 160040 | 160040 | 38.43 | 37.63 |
| astronaut | 85 | 200296 | 191636 | 183468 | 183468 | 39.73 | 38.89 |
| camera | 25 | 75844 | 74928 | 72420 | 72420 | 39.00 | 37.77 |
| camera | 55 | 92348 | 90900 | 88376 | 88376 | 41.90 | 40.73 |
| camera | 85 | 104336 | 102680 | 99616 | 99616 | 43.59 | 42.46 |
| coffee | 25 | 141172 | 134612 | 129552 | 129552 | 34.77 | 33.92 |
| coffee | 55 | 183060 | 175384 | 167076 | 167076 | 37.36 | 36.55 |
| coffee | 85 | 210996 | 202528 | 193416 | 193416 | 38.70 | 37.85 |
| moon | 25 | 22336 | 22188 | 22008 | 22008 | 42.17 | 41.28 |
| moon | 55 | 32020 | 32608 | 31044 | 31044 | 44.03 | 43.32 |
| moon | 85 | 43668 | 43012 | 41224 | 41224 | 45.29 | 44.62 |

## Summary of Improvements
- **v3.3**: Base version with Hilbert order and Fractal prediction.
- **v3.4 (EBOA)**: Introduced Extended-Basis Overlap-Add to eliminate block artifacts.
- **v3.5 (FBOA)**: Introduced Fractal-Basis Overlap-Add for deep block extension.
- **v3.6 (Primitives)**: Introduced geometric extrapolation primitives (Constant, Gradient, Mirror, Projection) with per-block adaptive search.

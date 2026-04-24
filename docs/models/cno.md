# CNO — Convolutional Neural Operator

**Paper:** Raonić et al., "Convolutional Neural Operators for robust and accurate learning of PDEs"
**Venue:** NeurIPS 2023
**arXiv:** https://arxiv.org/abs/2302.01178
**Code:** https://github.com/camlab-ethz/ConvolutionalNeuralOperator

## Key Idea

CNO is designed to preserve continuous-discrete equivalence: the discretized network approximates the same operator regardless of resolution. This avoids aliasing artifacts that FNO can suffer from. It achieves this via anti-aliased convolutions (sinc-based filters) applied in a U-Net-style encoder-decoder.

## Architecture

| Component | Value |
|-----------|-------|
| Channel progression | 17 → 64 → 128 → 256 |
| D/U blocks (down/upsampling) | 3 each |
| I blocks (integral/spectral) | 6 |
| R blocks (resampling) | 6 |
| Kernel size | 3 |
| Anti-aliasing filter taps (M) | 16 |
| Filter cutoff | f_s / 2.0001 |

## Model Size

~5.3 million parameters (as reported in the original paper experiments).

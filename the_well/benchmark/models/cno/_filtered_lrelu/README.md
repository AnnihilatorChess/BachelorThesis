# Vendored `filtered_lrelu` CUDA extension

This directory contains the compiled-CUDA implementation of the alias-free
activation used by the CNO model when `activation: compiled` is selected in the
model config. It is vendored verbatim (with two small patches, listed below)
from upstream so that training does not depend on an external repository being
cloned and on `PYTHONPATH`.

## Origin

- **Direct source:** [`camlab-ethz/ConvolutionalNeuralOperator`](https://github.com/camlab-ethz/ConvolutionalNeuralOperator),
  specifically `CNO2d_classic/torch_utils/` and `CNO2d_classic/dnnlib/`, as
  shipped with the NeurIPS 2023 paper *Convolutional Neural Operators for
  robust and accurate learning of PDEs* (Raonić et al.).
- **Upstream of that:** the same files were originally vendored by the CNO
  authors from NVIDIA's [`NVlabs/stylegan3`](https://github.com/NVlabs/stylegan3).
  The fused `filtered_lrelu` op (upsample → bias → leaky ReLU → clamp →
  downsample with Kaiser low-pass filters) is the StyleGAN3 alias-free
  primitive.

## License

`LICENSE_NVIDIA.txt` (NVIDIA Source Code License for StyleGAN3) — the same
license that ships with the upstream CNO repo. Non-commercial; research and
evaluation use only. Compatible with thesis use; **not redistributable for
commercial purposes**.

## What this directory provides

- `dnnlib/` — tiny utility module (`EasyDict`, path helpers).
- `torch_utils/` — `custom_ops.py` (JIT compile helper), `misc.py`,
  `persistence.py`.
- `torch_utils/ops/` — the actual extension:
  - `filtered_lrelu.{cpp,cu,h,py}` + `filtered_lrelu_{ns,rd,wr}.cu` —
    the fused alias-free LeakyReLU op.
  - `upfirdn2d.{cpp,cu,h,py}` — 2D upsample/FIR-filter/downsample primitive
    that `filtered_lrelu` depends on.
  - `bias_act.{cpp,cu,h,py}` — bias + activation helper (also a dep).
  - `conv2d_gradfix.py`, `conv2d_resample.py`, `grid_sample_gradfix.py`,
    `fma.py` — additional StyleGAN3 helpers brought along for completeness.

## How it is used

The vendored ops are loaded only when the model is constructed with
`activation="compiled"` (see `the_well.benchmark.models.cno.CompiledCNOActivation`).
The first time that activation is instantiated on a CUDA device, PyTorch's
`torch.utils.cpp_extension.load` invokes `ninja` + `nvcc` to compile the `.cu`
and `.cpp` sources. The resulting `.so` is cached in
`$TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions/`) and reused on
subsequent runs.

**System requirements (for the Linux training server):**

- CUDA toolkit with `nvcc` on `$PATH`, version matching `torch.version.cuda`.
- `gcc` ≥ 7 (any modern host compiler).
- `ninja` installed in the Python env.
- `scipy` (used by the activation wrapper for Kaiser filter design).

If any of these are missing, importing the activation will raise at module
load time. Use `activation="torch"` (the pure-PyTorch path, supports 1D / 2D /
3D) as a fallback.

## Patches vs. upstream

Two minor edits were applied to make the code work in a modern Python env
without changing behaviour:

1. `torch_utils/ops/conv2d_gradfix.py` — replaced
   `from pkg_resources import parse_version` with
   `from packaging.version import parse as parse_version`. The
   `pkg_resources` API was removed in `setuptools>=81`.
2. `torch_utils/ops/grid_sample_gradfix.py` — same patch as above.

All other files are verbatim copies of the upstream sources. The
package-level `__init__.py` (this directory) and `torch_utils/__init__.py` are
**additions** required because the upstream layout relies on implicit
namespace packages and absolute imports (`import dnnlib`,
`from torch_utils.ops import ...`); the package init aliases those names in
`sys.modules` so the vendored code resolves without edits to its own imports.

## Re-syncing from upstream

To pick up a new upstream release of CNO / StyleGAN3:

1. `git clone https://github.com/camlab-ethz/ConvolutionalNeuralOperator`
2. Copy `CNO2d_classic/torch_utils/` and `CNO2d_classic/dnnlib/` over this
   directory.
3. Re-apply the two `parse_version` patches.
4. Keep `__init__.py` (this dir) and `torch_utils/__init__.py` — they are
   ours, not upstream.
5. Re-run `scripts/bench_cno_compiled.py` to confirm the extension still
   compiles and matches the torch-path output.

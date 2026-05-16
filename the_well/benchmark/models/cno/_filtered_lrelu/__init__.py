# Vendored StyleGAN3 / CNO ops — see README.md in this directory for license/origin.
#
# The upstream code uses absolute imports `import dnnlib` and `from torch_utils...`
# without the parent package qualifier. To avoid editing the vendored sources (so
# we can re-sync them cleanly), we register aliases in sys.modules pointing at our
# nested packages. Any submodule imported via this package will run __init__.py
# first, so the aliases are in place before `import dnnlib` is encountered.

import sys as _sys

from . import dnnlib as _dnnlib
from . import torch_utils as _torch_utils

_sys.modules.setdefault("dnnlib", _dnnlib)
_sys.modules.setdefault("torch_utils", _torch_utils)
# torch_utils submodules referenced via `from torch_utils.ops import ...` resolve
# automatically once `torch_utils` itself is aliased, but we also register the
# .ops subpackage eagerly so `import torch_utils.ops.filtered_lrelu` works even
# if a downstream caller imports them out of order.
from .torch_utils import ops as _torch_utils_ops  # noqa: E402
_sys.modules.setdefault("torch_utils.ops", _torch_utils_ops)

"""FC-Former: the network (fcformer.model.fcformer) and task adapters (fcformer.tasks).

When a udl_vis task repo is already loaded (PanCollection or HyperSpectralCollection), importing fcformer also
registers the model `FCFormer` in it. The configs do this with `import_path: ["${hydra:runtime.cwd}", fcformer]`.
"""
import sys

from fcformer.model.fcformer import FCFormer

if "pancollection" in sys.modules:
    import fcformer.tasks.pansharpening  # noqa: F401
if "hisr" in sys.modules:
    import fcformer.tasks.hyperspectral  # noqa: F401

__all__ = ["FCFormer"]
__version__ = "1.0.0"

"""Helpers shared by the fusion passes: naming and loading generated kernels."""
import importlib.util
import os
import tempfile

import torch
import torch.fx as fx

KERNEL_DIR = "generated_kernels"
os.makedirs(KERNEL_DIR, exist_ok=True)

_kernel_counter = 0


def tensor_meta(node):
    """The fake tensor FX recorded for `node` (shape, dtype, ...), or None."""
    val = node.meta.get("val") if isinstance(node, fx.Node) else None
    return val if isinstance(val, torch.Tensor) else None


def new_kernel_name(prefix):
    global _kernel_counter
    _kernel_counter += 1
    return f"{prefix}_{_kernel_counter - 1}"


def load_kernel(name, source):
    # triton.jit needs the source to live in a real file (it uses inspect).
    path = os.path.join(KERNEL_DIR, f"{name}.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)

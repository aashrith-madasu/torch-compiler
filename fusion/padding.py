"""Pad the matmuls cuBLAS still runs when their sizes are bad for it.

cuBLAS falls back to slower kernels when K or N is not a multiple of 8 (fp16)
or 4 (fp32). We pad with zeros up to a good size, multiply, and return a slice
of the result. M barely matters, so it is left alone.

Run this after the fusion passes so it only sees the leftover matmuls.
"""
import os

import torch
import torch.fx as fx
import torch.nn.functional as F
import triton

from fusion.codegen import tensor_meta

aten = torch.ops.aten

ALIGNMENT = {torch.float16: 8, torch.bfloat16: 8, torch.float32: 4}

VIEW_OPS = {
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.flatten.using_ints,
}

# Whether padding won, per matmul shape and layout. Timed on first use.
_should_pad: dict[tuple, bool] = {}


def _round_up(x: int, alignment: int) -> int:
    return x + (-x % alignment)


def _pad_matrix(m: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a matrix at the end of each dim, keeping a transposed matrix
    transposed so the copy reads memory in order."""
    if m.stride(0) < m.stride(1):
        return F.pad(m.t(), (0, rows - m.shape[0], 0, cols - m.shape[1])).t()
    return F.pad(m, (0, cols - m.shape[1], 0, rows - m.shape[0]))


# ---------------------------------------------------------------- matching

def _keeps_last_dim(node: fx.Node, n: int) -> bool:
    """A padded result has gaps between rows, so it cannot be viewed in a way
    that merges the last dim with another. Follows chains of views."""
    for user in node.users:
        if user.target in VIEW_OPS:
            meta = tensor_meta(user)
            if meta is None or meta.shape[-1] != n or not _keeps_last_dim(user, n):
                return False
    return True


def _should_try_padding(node: fx.Node) -> bool:
    if node.op != "call_function" or node.kwargs:  # beta / alpha other than 1
        return False
    if node.target == aten.addmm.default:
        bias, a, b = (tensor_meta(x) for x in node.args)
        if bias is None or a is None or b is None or bias.shape[-1:] != b.shape[-1:]:
            return False
    elif node.target == aten.mm.default:
        a, b = (tensor_meta(x) for x in node.args)
        if a is None or b is None:
            return False
    else:
        return False

    alignment = ALIGNMENT.get(a.dtype)
    if alignment is None or a.dtype != b.dtype or not (a.is_cuda and b.is_cuda):
        return False
    shape = (*a.shape, *b.shape)
    if not all(isinstance(s, int) for s in shape):  # static shapes only
        return False
    k, n = a.shape[1], b.shape[1]
    if k % alignment == 0 and n % alignment == 0:
        return False
    return _keeps_last_dim(node, n)


# ----------------------------------------------------------------- rewrite

def build_padded_op(target, alignment):
    def plain(*args):
        return target(*args)

    def padded(*args):
        *bias, a, b = args
        m, k = a.shape
        n = b.shape[1]
        k_pad, n_pad = _round_up(k, alignment), _round_up(n, alignment)
        pad_args = [
            *(F.pad(x, (0, n_pad - n)) for x in bias),
            _pad_matrix(a, m, k_pad),
            _pad_matrix(b, k_pad, n_pad),
        ]
        return target(*pad_args)[:, :n]

    def padded_op(*args):
        a, b = args[-2:]
        key = (target, a.shape, b.shape, a.dtype, a.stride(), b.stride())
        if key not in _should_pad:
            plain_ms = triton.testing.do_bench(lambda: plain(*args), warmup=2, rep=5)
            padded_ms = triton.testing.do_bench(lambda: padded(*args), warmup=2, rep=5)
            _should_pad[key] = padded_ms < plain_ms
            if os.getenv("KERNEL_FUSION_DEBUG"):
                print(f"[padding] {a.shape} x {b.shape}: plain {plain_ms:.3f} ms, "
                      f"padded {padded_ms:.3f} ms")
        return padded(*args) if _should_pad[key] else plain(*args)

    padded_op.__name__ = f"padded_{target.name().removeprefix('aten::')}"
    return padded_op


def pad_matmuls(gm: fx.GraphModule) -> fx.GraphModule:
    padded = []
    for node in list(gm.graph.nodes):
        if not _should_try_padding(node):
            continue
        alignment = ALIGNMENT[tensor_meta(node.args[-2]).dtype]
        padded_op = build_padded_op(node.target, alignment)
        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_function(padded_op, node.args)
        new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        padded.append(new_node.name)

    if os.getenv("KERNEL_FUSION_DEBUG"):
        print("========== Matmul padding =============")
        print(f"may pad {len(padded)} matmuls: {', '.join(padded)}")

    gm.graph.lint()
    gm.recompile()
    return gm

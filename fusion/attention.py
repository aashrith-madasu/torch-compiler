"""Use the fast flash attention kernel when the attention mask is just causal.

While compiling, Hugging Face builds an explicit causal mask instead of passing
is_causal=True, which forces the slower efficient attention kernel. If the mask
is a constant that blocks exactly the future positions, flash attention with
is_causal=True computes the same thing.
"""
import operator
import os

import torch
import torch.fx as fx
from torch.fx.node import map_arg

from fusion.codegen import tensor_meta

aten = torch.ops.aten
EFFICIENT = aten._scaled_dot_product_efficient_attention.default
FLASH = aten._scaled_dot_product_flash_attention.default

# A float mask is added to the attention scores, so anything this negative
# zeroes that position after softmax.
BLOCKED = -1e4


class _NotConstant(Exception):
    pass


def _evaluate(node: fx.Node, cache: dict):
    """Compute a node's value now. Raises _NotConstant if it depends on a
    graph input, like a padding mask passed in by the caller."""
    if node not in cache:
        if node.op != "call_function":
            raise _NotConstant
        args = map_arg(node.args, lambda n: _evaluate(n, cache))
        kwargs = map_arg(node.kwargs, lambda n: _evaluate(n, cache))
        cache[node] = node.target(*args, **kwargs)
    return cache[node]


def _is_causal_mask(mask: torch.Tensor, seq_len: int) -> bool:
    if not mask.is_floating_point() or mask.shape[-2:] != (seq_len, seq_len):
        return False
    # Dims added by expand share memory, so one index covers them. The rest
    # are checked in full, since each batch element may have its own copy.
    index = tuple(0 if mask.stride(d) == 0 else slice(None) for d in range(mask.ndim - 2))
    m = mask[index]
    future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=m.device).triu(1)
    return bool((m[..., future] <= BLOCKED).all() and (m[..., ~future] == 0).all())


def _can_rewrite(node: fx.Node, cache: dict) -> bool:
    if len(node.args) != 5 or not isinstance(node.args[3], fx.Node):
        return False
    kwargs = dict(node.kwargs)
    if kwargs.pop("dropout_p", 0.0) != 0.0 or kwargs.pop("is_causal", False) or set(kwargs) - {"scale"}:
        return False
    # Only the output tensor may be used, since the other results differ.
    if not all(u.target is operator.getitem and u.args[1] == 0 for u in node.users):
        return False

    q, k = tensor_meta(node.args[0]), tensor_meta(node.args[1])
    if q is None or k is None or q.shape[-2] != k.shape[-2]:
        return False
    try:
        # Fake tensors are on during compile; we need real values here.
        with torch.utils._python_dispatch._disable_current_modes():
            return _is_causal_mask(_evaluate(node.args[3], cache), q.shape[-2])
    except _NotConstant:
        return False


def build_flash_op(scale):
    def flash_attention(q, k, v):
        out = FLASH(q, k, v, is_causal=True, scale=scale)[0]
        # Later views assume efficient attention's memory order ([B, S, H, D]
        # even though the shape is [B, H, S, D]), so match it.
        return out.transpose(1, 2).contiguous().transpose(1, 2)

    return flash_attention


def use_flash_attention(gm: fx.GraphModule) -> fx.GraphModule:
    rewritten = []
    cache = {}
    with torch.no_grad():
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target != EFFICIENT or not _can_rewrite(node, cache):
                continue
            flash_op = build_flash_op(node.kwargs.get("scale"))
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(flash_op, tuple(node.args[:3]))
            for getitem in list(node.users):
                new_node.meta = dict(getitem.meta)
                getitem.replace_all_uses_with(new_node)
                gm.graph.erase_node(getitem)
            gm.graph.erase_node(node)
            rewritten.append(new_node.name)

    if rewritten:
        gm.graph.eliminate_dead_code()  # the mask is no longer needed

    if os.getenv("KERNEL_FUSION_DEBUG"):
        print("========== Flash attention =============")
        print(f"rewrote {len(rewritten)} attention calls: {', '.join(rewritten)}")

    gm.graph.lint()
    gm.recompile()
    return gm

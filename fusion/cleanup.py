"""Graph cleanups that run before any fusion pass."""
import os
import torch
import torch.fx as fx

from fusion.codegen import tensor_meta

aten = torch.ops.aten

# Ops checked to give identical results whatever the layout of their inputs.
# Only add an op here after comparing it against packed (contiguous) inputs.
LAYOUT_AGNOSTIC_OPS = {
    aten._scaled_dot_product_efficient_attention.default,
}


def _is_redundant_clone(node: fx.Node) -> bool:
    """The graph never writes to a tensor in place, so a clone only matters if
    its result leaves the graph or it changes the memory layout."""
    if node.op != "call_function" or node.target != aten.clone.default:
        return False
    if any(user.op == "output" for user in node.users):
        return False
    src, dst = tensor_meta(node.args[0]), tensor_meta(node)
    if src is None or dst is None:
        return False
    return src.stride() == dst.stride() or all(
        user.target in LAYOUT_AGNOSTIC_OPS for user in node.users
    )


def remove_redundant_clones(gm: fx.GraphModule) -> fx.GraphModule:
    removed = []
    for node in list(gm.graph.nodes):
        if _is_redundant_clone(node):
            removed.append(node.name)
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)

    if os.getenv("KERNEL_FUSION_DEBUG"):
        print("========== Clone cleanup =============")
        print(f"removed {len(removed)} clones: {', '.join(removed)}")

    gm.graph.lint()
    gm.recompile()
    return gm

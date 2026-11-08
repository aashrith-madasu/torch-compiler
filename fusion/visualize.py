import torch
import torch.fx as fx


def _op_name(target) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)  # e.g. "mul.Tensor"
    return getattr(target, "__name__", str(target))  # e.g. "fused_3"


def _arg(a) -> str:
    if isinstance(a, fx.Node):
        return f"%{a.name}"
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(map(_arg, a)) + "]"
    return repr(a)


def visualize(graph: fx.Graph, title: str = "graph"):
    """Print one line per node: name = op(args)   shape dtype."""
    print(f"\n===== {title} =====")
    for node in graph.nodes:
        if node.op == "placeholder":
            text = f"%{node.name} = input"
        elif node.op == "get_attr":
            text = f"%{node.name} = param {node.target}"
        elif node.op == "output":
            text = f"return {_arg(node.args[0])}"
        else:
            text = f"%{node.name} = {_op_name(node.target)}({', '.join(map(_arg, node.args))})"

        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            text = f"{text:<60} {list(val.shape)} {str(val.dtype).removeprefix('torch.')}"
        print(text)

"""Reduction fusion: a reduction (mean, sum, amax or amin) over the last dim
plus the pointwise ops around it becomes one Triton kernel.

RMSNorm is the main target:

    copy = _to_copy(x)            \
    p    = pow(copy, 2)            | loop 1: sum over the last dim
    m    = mean(p, [-1], True)    /
    a    = add(m, eps)            \ per-row values, computed once between loops
    r    = rsqrt(a)               /
    y    = mul(copy, r)           \
    z    = _to_copy(y)             | loop 2: every element again, then store
    out  = mul(weight, z)         /

Only `out` is written to memory. Run this after fuse_epilogue, which gets first
pick of ops after a matmul, and before fuse_pointwise, which takes the rest.
"""
import os
from typing import NamedTuple

import torch
import torch.fx as fx
import triton

from fusion.codegen import load_kernel, new_kernel_name, tensor_meta
from fusion.pointwise import POINTWISE_OPS, _is_fusible, external_inputs

aten = torch.ops.aten

ROUNDED_DTYPES = {torch.float16: "tl.float16", torch.bfloat16: "tl.bfloat16"}

AUTOTUNE_CONFIGS = """[
        triton.Config({"XBLOCK": 1, "RBLOCK": 1024}, num_warps=4),
        triton.Config({"XBLOCK": 4, "RBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 8, "RBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 16, "RBLOCK": 128}, num_warps=4),
        triton.Config({"XBLOCK": 32, "RBLOCK": 64}, num_warps=4),
    ]"""


class Reduction(NamedTuple):
    """Triton code for one reduction. `acc` is an [XBLOCK, RBLOCK] tile."""
    init: str  # starting value of `acc`
    step: str  # new `acc` after seeing the tile `{x}`
    finish: str  # one value per row, shaped [XBLOCK, 1]; `{n}` is the row length
    helper: str = ""  # extra Triton functions the kernel needs


def _extremum(tl_fn: str, start: str) -> Reduction:
    """A max or min that keeps NaNs, like torch.amax and torch.amin."""
    combine = f"_reduce_{tl_fn}"
    return Reduction(
        init=f"tl.full((XBLOCK, RBLOCK), {start}, tl.float32)",
        step=f"tl.{tl_fn}(acc, tl.where(rmask, {{x}}, {start}), tl.PropagateNan.ALL)",
        finish=f"tl.reduce(acc, 1, {combine})[:, None]",
        helper=f"@triton.jit\ndef {combine}(a, b):\n    return tl.{tl_fn}(a, b, tl.PropagateNan.ALL)\n\n",
    )


_ZEROS = "tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)"
_ADD = "acc + tl.where(rmask, {x}, 0.0)"

REDUCTIONS = {
    aten.mean.dim: Reduction(_ZEROS, _ADD, "tl.sum(acc, 1)[:, None] / {n}"),
    aten.sum.dim_IntList: Reduction(_ZEROS, _ADD, "tl.sum(acc, 1)[:, None]"),
    aten.amax.default: _extremum("maximum", "-1e999"),  # 1e999 is infinity
    aten.amin.default: _extremum("minimum", "1e999"),
}

# Generated kernels keyed by their source (with the name left out), so
# identical layers share one kernel and autotune only once.
_kernel_cache: dict[str, tuple[str, object]] = {}


# ---------------------------------------------------------------- matching

def _is_float_cast(node: fx.Node) -> bool:
    """Any float-to-float cast up to fp32. Unlike the pointwise pass, this one
    can fuse past a cast to fp16 because _emit does the rounding explicitly."""
    if node.op != "call_function" or node.target != aten._to_copy.default:
        return False
    if "dtype" not in node.kwargs or set(node.kwargs) - {"dtype", "layout", "device"}:
        return False
    if node.kwargs.get("layout", torch.strided) != torch.strided:
        return False
    src, dst = tensor_meta(node.args[0]), tensor_meta(node)
    return (
        src is not None
        and dst is not None
        and dst.is_cuda
        and src.device == dst.device
        and src.is_floating_point()
        and dst.is_floating_point()
        and torch.float64 not in (src.dtype, dst.dtype)
    )


def _can_fuse(node: fx.Node) -> bool:
    return _is_fusible(node) or _is_float_cast(node)


def _is_reduction(node: fx.Node) -> bool:
    """A reduction over [last dim] with keepdim=True and a static, float shape."""
    if node.op != "call_function" or node.target not in REDUCTIONS:
        return False
    if len(node.args) != 3 or node.kwargs or node.args[2] is not True:
        return False
    x, dims = tensor_meta(node.args[0]), node.args[1]
    if x is None or not x.is_cuda or not x.is_floating_point() or x.ndim < 2:
        return False
    if not all(isinstance(s, int) for s in x.shape) or x.shape[-1] == 1:
        return False
    return isinstance(dims, (list, tuple)) and len(dims) == 1 and dims[0] % x.ndim == x.ndim - 1


def _ancestors(start: list[fx.Node], members: set[fx.Node]) -> set[fx.Node]:
    """`start` and everything in `members` that feeds it."""
    seen, stack = set(), [n for n in start if n in members]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack += [a for a in node.all_input_nodes if a in members]
    return seen


def _collect(reduction: fx.Node, order: dict) -> tuple[list[fx.Node], fx.Node] | None:
    """Find the kernel around `reduction`. Returns (nodes in graph order, root),
    where the root is the only node read from outside the kernel."""
    full = tensor_meta(reduction.args[0]).shape
    per_row = tensor_meta(reduction).shape

    # Everything fusible downstream of the reduction, at either shape.
    post, frontier = set(), list(reduction.users)
    while frontier:
        node = frontier.pop()
        if node not in post and _can_fuse(node) and tensor_meta(node).shape in (full, per_row):
            post.add(node)
            frontier += list(node.users)

    # Drop nodes from the end until exactly one result leaves the kernel.
    candidates = sorted(post, key=order.__getitem__)
    root = None
    while candidates:
        members = {reduction, *candidates}
        leaving = [n for n in members if any(u not in members for u in n.users)]
        if leaving == [candidates[-1]] and tensor_meta(candidates[-1]).shape == full:
            root = candidates[-1]
            break
        candidates.pop()
    if root is None:
        return None

    # Per-row values are computed between the loops, so they can only depend
    # on the reduction and on each other.
    per_row_nodes = {n for n in candidates if tensor_meta(n).shape == per_row}
    if any(a not in per_row_nodes and a is not reduction for n in per_row_nodes for a in n.all_input_nodes):
        return None

    # Pull in fusible producers that only this kernel reads. Going backwards
    # means a node's users are settled before the node is considered.
    for node in sorted(order, key=order.__getitem__, reverse=True):
        if (
            order[node] < order[root]
            and node not in members
            and _can_fuse(node)
            and tensor_meta(node).shape == full
            and node.users
            and all(u in members for u in node.users)
        ):
            members.add(node)
    return sorted(members, key=order.__getitem__), root


# ----------------------------------------------------------------- codegen

def _emit(nodes, names, lines, prefix, indent):
    """Append one Triton line per node, using variables `prefix0`, `prefix1`..."""
    def operand(arg):
        return names[arg] if isinstance(arg, fx.Node) else repr(float(arg))

    for j, node in enumerate(nodes):
        tl_dtype = ROUNDED_DTYPES.get(tensor_meta(node).dtype)
        if node.target == aten._to_copy.default and tl_dtype is not None:
            # Values stay in fp32 inside the kernel, so round here to match
            # what eager computes after this cast.
            expr = f"{operand(node.args[0])}.to({tl_dtype}).to(tl.float32)"
        else:
            expr = POINTWISE_OPS[node.target](*map(operand, node.args))
        lines.append(f"{indent}{prefix}{j} = {expr}  # {node.target}")
        names[node] = f"{prefix}{j}"


def generate_kernel_source(name, nodes, reduction, root, inputs):
    members = set(nodes)
    rank = tensor_meta(root).ndim
    rnumel = tensor_meta(reduction.args[0]).shape[-1]
    per_row = tensor_meta(reduction).shape

    first = _ancestors([reduction.args[0]], members)
    per_row_nodes = [n for n in nodes if n is not reduction and tensor_meta(n).shape == per_row]
    stop = {reduction, *per_row_nodes}
    second = _ancestors([root], members - stop)
    loop1 = [n for n in nodes if n in first]
    loop2 = [n for n in nodes if n in second]

    in_ptrs = [f"in{i}_ptr" for i in range(len(inputs))]
    sizes = [f"size{d}" for d in range(rank - 1)]
    strides = [f"in{i}_s{d}" for i in range(len(inputs)) for d in range(rank)]
    params = ", ".join(
        in_ptrs + ["out_ptr", "xnumel", "rnumel"] + sizes + strides
        + ["XBLOCK: tl.constexpr", "RBLOCK: tl.constexpr"]
    )

    op = REDUCTIONS[reduction.target]
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        op.helper,
        f"@triton.autotune(configs={AUTOTUNE_CONFIGS}, key=['xnumel', 'rnumel'])",
        "@triton.jit",
        f"def {name}({params}):",
        "    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]",
        "    rbase = tl.arange(0, RBLOCK)[None, :]",
        "    xmask = xindex < xnumel",
        "    rem = xindex",
    ]
    for d in reversed(range(rank - 1)):
        if d == 0:
            lines.append("    idx0 = rem")
        else:
            lines.append(f"    idx{d} = rem % size{d}")
            lines.append(f"    rem = rem // size{d}")

    def open_loop(prefix, used_nodes, extra=()):
        """Start a loop over the last dim and load the inputs it uses."""
        lines.append("    for roffset in range(0, rnumel, RBLOCK):")
        lines.append("        rindex = roffset + rbase")
        lines.append("        rmask = (rindex < rnumel) & xmask")
        names = {}
        used = {*external_inputs(used_nodes), *extra} - members
        for i, inp in enumerate(inputs):
            if inp in used:
                offset = " + ".join(
                    [f"idx{d} * in{i}_s{d}" for d in range(rank - 1)]
                    + [f"rindex * in{i}_s{rank - 1}"]
                )
                lines.append(
                    f"        {prefix}in{i} = tl.load(in{i}_ptr + {offset}, mask=rmask, other=0.0)"
                    ".to(tl.float32)"
                )
                names[inp] = f"{prefix}in{i}"
        return names

    # Loop 1: accumulate the reduction over the last dim.
    lines.append(f"    acc = {op.init}")
    names = open_loop("a", loop1, extra=[reduction.args[0]])
    _emit(loop1, names, lines, "p", "        ")
    lines.append(f"        acc = {op.step.format(x=names[reduction.args[0]])}")

    # Per-row values: [XBLOCK, 1], broadcast against the tiles in loop 2.
    names = {reduction: "reduced"}
    finish = op.finish.format(n=repr(float(rnumel)))
    tl_dtype = ROUNDED_DTYPES.get(tensor_meta(reduction).dtype)
    if tl_dtype is not None:  # eager's result is rounded before later ops see it
        finish = f"({finish}).to({tl_dtype}).to(tl.float32)"
    lines.append(f"    reduced = {finish}")
    _emit(per_row_nodes, names, lines, "s", "    ")

    # Loop 2: compute every element and store the root.
    names.update(open_loop("b", loop2))
    _emit(loop2, names, lines, "q", "        ")
    lines.append(f"        tl.store(out_ptr + xindex * rnumel + rindex, {names[root]}, mask=rmask)")
    return "\n".join(lines) + "\n"


def get_kernel(nodes, reduction, root, inputs):
    key = generate_kernel_source("KERNEL", nodes, reduction, root, inputs)
    if key not in _kernel_cache:
        name = new_kernel_name("fused_reduction")
        source = generate_kernel_source(name, nodes, reduction, root, inputs)
        _kernel_cache[key] = (name, load_kernel(name, source))
    return _kernel_cache[key]


def build_fused_op(nodes, reduction, root, inputs):
    name, kernel = get_kernel(nodes, reduction, root, inputs)
    out_meta = tensor_meta(root)
    shape, dtype = tuple(out_meta.shape), out_meta.dtype
    rnumel = shape[-1]

    def fused_op(*tensors):
        out = torch.empty(shape, dtype=dtype, device=tensors[0].device)
        xnumel = out.numel() // rnumel
        strides = [s for t in tensors for s in t.expand(shape).stride()]
        grid = lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]),)
        kernel[grid](*tensors, out, xnumel, rnumel, *shape[:-1], *strides)
        return out

    fused_op.__name__ = name
    return fused_op


# -------------------------------------------------------------------- pass

def fuse_reduction(gm: fx.GraphModule) -> fx.GraphModule:
    debug = os.getenv("KERNEL_FUSION_DEBUG")
    if debug:
        print("========== Reduction fusion =============")

    for reduction in [n for n in gm.graph.nodes if _is_reduction(n)]:
        order = {n: i for i, n in enumerate(gm.graph.nodes)}
        found = _collect(reduction, order)
        if found is None:
            continue
        nodes, root = found
        inputs = external_inputs(nodes)
        fused_op = build_fused_op(nodes, reduction, root, inputs)
        members = ",".join(n.name for n in nodes)

        with gm.graph.inserting_before(root):
            new_node = gm.graph.call_function(fused_op, tuple(inputs))
        new_node.meta = dict(root.meta)
        root.replace_all_uses_with(new_node)
        for n in reversed(nodes):  # consumers before producers
            gm.graph.erase_node(n)

        if debug:
            print(f"{members} --> {fused_op.__name__}"
                  f"   (reads: {', '.join(n.name for n in inputs)})")

    gm.graph.lint()
    gm.recompile()
    return gm

"""Epilogue fusion: addmm (+ optional view) followed by a pointwise group
becomes one Triton matmul that runs the pointwise ops on each output tile
before storing it.

    addmm_1 = addmm(bias, A, B)              [M, N]
    view_9  = view(addmm_1, [32, 1024, N])   (optional, only splits the rows)
    clone_4 = clone(view_9)                  \
    add_2   = add(clone_4, residual)         / pointwise group
      -->
    fused_gemm_0(A, B, bias, residual)

The addmm result never goes to memory. Other inputs of the group (residual,
broadcast scales, ...) are loaded as [BLOCK_M, BLOCK_N] tiles. Run this pass
before fuse_pointwise so it gets first pick of the groups.
"""
import os
import torch
import torch.fx as fx
import triton

from fusion.codegen import load_kernel, new_kernel_name, tensor_meta
from fusion.pointwise import emit_pointwise_ops, external_inputs, find_fusion_groups

aten = torch.ops.aten

# Must fit the ~100 KB of shared memory per SM on Ampere consumer/workstation
# GPUs: (BLOCK_M + BLOCK_N) * BLOCK_K * 2 bytes * num_stages.
AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ]"""

# Generated kernels keyed by their source (with the name left out), so the 12
# identical layers of a transformer share one kernel and autotune only once.
_kernel_cache: dict[str, tuple[str, object]] = {}


# ---------------------------------------------------------------- matching

def match_addmm(group: list[fx.Node]):
    """Return (addmm, view_or_None, input_node) if one of the group's inputs
    is an addmm whose result is used only by this group, else None."""
    members = set(group)
    group_shape = tensor_meta(group[0]).shape

    for inp in external_inputs(group):
        mm, view = inp, None
        if inp.target == aten.view.default:
            mm, view = inp.args[0], inp

        if not (isinstance(mm, fx.Node) and mm.target == aten.addmm.default):
            continue
        if mm.kwargs:  # beta / alpha other than 1
            continue
        bias, a, b = (tensor_meta(x) for x in mm.args)
        out = tensor_meta(mm)
        if None in (bias, a, b, out):
            continue
        if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
            continue

        # The addmm result must not be needed anywhere else, or we would have
        # to write it to memory anyway.
        if view is None:
            if not all(u in members for u in mm.users):
                continue
        elif list(mm.users) != [view] or not all(u in members for u in view.users):
            continue

        # The group works on the addmm result itself, not a broadcast of it,
        # and a view may only split the rows: [M, N] -> [..., N].
        if tensor_meta(inp).shape != group_shape or group_shape[-1] != out.shape[1]:
            continue
        return mm, view, inp
    return None


# ----------------------------------------------------------------- codegen

def generate_kernel_source(name, group, mm_input, others):
    """Matmul over [M, K] x [K, N] with the group's ops applied to each
    output tile. `others` are the group's other inputs, each passed in as an
    [M, N] view (stride 0 on broadcast dims)."""
    ptrs = [f"in{i}_ptr" for i in range(len(others))]
    strides = [f"in{i}_s{d}" for i in range(len(others)) for d in range(2)]
    params = ", ".join(
        ["a_ptr", "b_ptr", "bias_ptr", *ptrs, "out_ptr", "M", "N", "K",
         "stride_am", "stride_ak", "stride_bk", "stride_bn", "bias_s0", "bias_s1", *strides,
         "BLOCK_M: tl.constexpr", "BLOCK_N: tl.constexpr", "BLOCK_K: tl.constexpr",
         "GROUP_M: tl.constexpr"]
    )

    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        f"@triton.autotune(configs={AUTOTUNE_CONFIGS}, key=['M', 'N', 'K'])",
        "@triton.jit",
        f"def {name}({params}):",
        # Grouped tile order: neighbouring programs share A rows in L2.
        "    pid = tl.program_id(0)",
        "    num_pid_m = tl.cdiv(M, BLOCK_M)",
        "    num_pid_n = tl.cdiv(N, BLOCK_N)",
        "    group_id = pid // (GROUP_M * num_pid_n)",
        "    first_pid_m = group_id * GROUP_M",
        "    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)",
        "    pid_m = first_pid_m + (pid % (GROUP_M * num_pid_n)) % group_size_m",
        "    pid_n = (pid % (GROUP_M * num_pid_n)) // group_size_m",
        "",
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        "    offs_k = tl.arange(0, BLOCK_K)",
        "    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak",
        "    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn",
        "",
        "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
        "    for k in range(0, tl.cdiv(K, BLOCK_K)):",
        "        k_rem = K - k * BLOCK_K",
        "        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)",
        "        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)",
        "        acc += tl.dot(a, b)",
        "        a_ptrs += BLOCK_K * stride_ak",
        "        b_ptrs += BLOCK_K * stride_bk",
        "",
        "    # ---- epilogue",
        "    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
        "    bias = tl.load(bias_ptr + offs_m[:, None] * bias_s0 + offs_n[None, :] * bias_s1, mask=mask)",
        "    mm = acc + bias.to(tl.float32)",
    ]

    names: dict[fx.Node, str] = {mm_input: "mm"}
    for i, inp in enumerate(others):
        lines.append(
            f"    x{i} = tl.load(in{i}_ptr + offs_m[:, None] * in{i}_s0"
            f" + offs_n[None, :] * in{i}_s1, mask=mask).to(tl.float32)"
        )
        names[inp] = f"x{i}"

    result = emit_pointwise_ops(group, names, lines)
    lines.append(f"    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], {result}, mask=mask)")
    return "\n".join(lines) + "\n"


def get_kernel(group, mm_input, others):
    """Build (or reuse) the kernel for this group. Returns (name, kernel)."""
    key = generate_kernel_source("KERNEL", group, mm_input, others)
    if key not in _kernel_cache:
        name = new_kernel_name("fused_gemm")
        source = generate_kernel_source(name, group, mm_input, others)
        _kernel_cache[key] = (name, load_kernel(name, source))
    return _kernel_cache[key]


def build_fused_op(group, mm_input, others):
    name, kernel = get_kernel(group, mm_input, others)
    out_meta = tensor_meta(group[-1])
    shape, dtype = tuple(out_meta.shape), out_meta.dtype

    def fused_op(bias, a, b, *tensors):  # same order as addmm(bias, a, b)
        M, K = a.shape
        N = b.shape[1]
        out = torch.empty(shape, dtype=dtype, device=a.device)
        bias = bias.expand(M, N)
        # View each input as [M, N]. This is free for contiguous and
        # broadcast inputs; reshape only copies for odd layouts.
        tiles = [t.expand(shape).reshape(M, N) for t in tensors]
        strides = [s for t in tiles for s in t.stride()]
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        kernel[grid](
            a, b, bias, *tiles, out, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            bias.stride(0), bias.stride(1), *strides,
        )
        return out

    fused_op.__name__ = name
    return fused_op


# -------------------------------------------------------------------- pass

def fuse_epilogue(gm: fx.GraphModule) -> fx.GraphModule:
    if os.getenv("FUSION_KERNEL_DEBUG"):
        print("========== Epilogue fusion =============")
        
    for group in find_fusion_groups(gm.graph):
        match = match_addmm(group)
        if match is None:
            continue
        mm, view, mm_input = match
        others = [n for n in external_inputs(group) if n is not mm_input]
        fused_op = build_fused_op(group, mm_input, others)

        root = group[-1]
        with gm.graph.inserting_before(root):
            new_node = gm.graph.call_function(fused_op, (*mm.args, *others))
        new_node.meta = dict(root.meta)
        root.replace_all_uses_with(new_node)
        for n in reversed(group):  # consumers before producers
            gm.graph.erase_node(n)
        if view is not None:
            gm.graph.erase_node(view)
        gm.graph.erase_node(mm)

        if os.getenv("FUSION_KERNEL_DEBUG"):
            print(f"{mm.name},{','.join(n.name for n in group)} --> {fused_op.__name__}"
                f"   (reads: {', '.join(n.name for n in new_node.args)})")

    gm.graph.lint()
    gm.recompile()
    return gm

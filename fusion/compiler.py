import os
import torch.fx as fx
from torch._functorch.aot_autograd import make_boxed_func

from fusion.attention import use_flash_attention
from fusion.cleanup import remove_redundant_clones
from fusion.epilogue import fuse_epilogue
from fusion.padding import pad_matmuls
from fusion.pointwise import fuse_pointwise
from fusion.reduction import fuse_reduction
from fusion.visualize import visualize

# Run in order. Cleanup goes first so the fusion passes see a simpler graph.
# Epilogue then gets first pick of the pointwise groups that follow a matmul;
# fuse_pointwise takes whatever is left. Padding goes last so it only sees the
# matmuls still going to cuBLAS.

def fusion_compiler(gm: fx.GraphModule, example_inputs):
    if os.getenv("KERNEL_FUSION_DEBUG"):
        visualize(gm.graph, "before fusion")
    gm = remove_redundant_clones(gm)
    gm = use_flash_attention(gm)  # after cleanup, which knows efficient attention
    gm = fuse_epilogue(gm)
    gm = fuse_reduction(gm)
    gm = fuse_pointwise(gm)
    gm = pad_matmuls(gm)
    if os.getenv("KERNEL_FUSION_DEBUG"):
        visualize(gm.graph, "after fusion")
    return make_boxed_func(gm.forward)

import os
import torch.fx as fx
from torch._functorch.aot_autograd import make_boxed_func

from fusion.cleanup import remove_redundant_clones
from fusion.epilogue import fuse_epilogue
from fusion.pointwise import fuse_pointwise
from fusion.visualize import visualize

# Run in order. Cleanup goes first so the fusion passes see a simpler graph.
# Epilogue then gets first pick of the pointwise groups that follow a matmul;
# fuse_pointwise takes whatever is left.

def fusion_compiler(gm: fx.GraphModule, example_inputs):
    if os.getenv("FUSION_KERNEL_DEBUG"):
        visualize(gm.graph, "before fusion")
    gm = remove_redundant_clones(gm)
    gm = fuse_epilogue(gm)
    gm = fuse_pointwise(gm)
    if os.getenv("FUSION_KERNEL_DEBUG"):
        visualize(gm.graph, "after fusion")
    return make_boxed_func(gm.forward)

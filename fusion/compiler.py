import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func

from fusion.pointwise import fuse_pointwise
from fusion.visualize import visualize

def fusion_compiler(gm: fx.GraphModule, example_inputs):
    # visualize(gm.graph, "before fusion")
    gm = fuse_pointwise(gm)
    # visualize(gm.graph, "after fusion")
    return make_boxed_func(gm.forward)
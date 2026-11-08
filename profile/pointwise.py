import triton
import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func

from fusion.compiler import fusion_compiler


def my_pointwise_math(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.tanh(x * 10))


my_pointwise_math_fused = torch.compile(my_pointwise_math, 
                                        backend=aot_autograd(fw_compiler=fusion_compiler))


if __name__ == "__main__":
    
    x = torch.randn(1024 * 10, 1024 * 10, dtype=torch.float16, device="cuda")
 
    # # ======= NSys Profiling ==============
    # # warm up
    # for _ in range(10):
    #     my_pointwise_math(x)
    #     my_pointwise_math_fused(x)
    
    # torch.cuda.synchronize()
    
    # torch.cuda.profiler.start()
    
    # with torch.cuda.nvtx.range("eager"):
    #     for _ in range(10):
    #         my_pointwise_math(x)
    #     torch.cuda.synchronize()
        
    # with torch.cuda.nvtx.range("fused"):
    #     for _ in range(10):
    #         my_pointwise_math_fused(x)
    #     torch.cuda.synchronize()
        
    # torch.cuda.profiler.stop()  
    
    # ======= Triton Profiling ==============
    ms = triton.testing.do_bench(lambda: my_pointwise_math(x))
    print(f"eager: {ms:.3f} ms")
    ms = triton.testing.do_bench(lambda: my_pointwise_math_fused(x))
    print(f"fused: {ms:.3f} ms")


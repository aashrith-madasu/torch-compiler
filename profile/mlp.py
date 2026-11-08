import triton
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from fusion.compiler import fusion_compiler



class MLPBlock(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        y = self.fc2(h)
        return x + torch.tanh(y) * self.scale
    

if __name__ == "__main__":
    
    model = MLPBlock().to(device="cuda", dtype=torch.float16)
    model_fused = torch.compile(model, 
                                backend=aot_autograd(fw_compiler=fusion_compiler))
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float16, requires_grad=False)
    
    with torch.no_grad():
        # # ======= NSys Profiling ==============
        # # warm up
        # for _ in range(10):
        #     model(x)
        #     model_fused(x)
        
        # torch.cuda.synchronize()
        
        # torch.cuda.profiler.start()
        
        # with torch.cuda.nvtx.range("eager"):
        #     for _ in range(10):
        #         model(x)
        #     torch.cuda.synchronize()
            
        # with torch.cuda.nvtx.range("fused"):
        #     for _ in range(10):
        #         model_fused(x)
        #     torch.cuda.synchronize()
            
        # torch.cuda.profiler.stop()  
    
        # ======= Triton Profiling ==============
        ms = triton.testing.do_bench(lambda: model(x))
        print(f"eager: {ms:.3f} ms")
        ms = triton.testing.do_bench(lambda: model_fused(x))
        print(f"fused: {ms:.3f} ms")
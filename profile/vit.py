import timm
import triton 
import torch
from torch._dynamo.backends.common import aot_autograd
from fusion.compiler import fusion_compiler

if __name__ == "__main__":

    model = timm.create_model(model_name='vit_base_patch16_224', 
                              pretrained=False, 
                              dtype=torch.float16,
                              device="cuda")
    model.eval()
    x = torch.randn(32, 3, 224, 224, 
                    dtype=torch.float16, device="cuda")
    model_fused = torch.compile(model, 
                                backend=aot_autograd(fw_compiler=fusion_compiler))

    with torch.no_grad():
        eager_ms = triton.testing.do_bench(lambda: model(x))
        fused_ms = triton.testing.do_bench(lambda: model_fused(x))
        
        print(
            f"\neager: {eager_ms:.3f} ms"
            f"\nfused: {fused_ms:.3f} ms"
            f"\nspeed up: {eager_ms/fused_ms:.3f}x"
        )
        
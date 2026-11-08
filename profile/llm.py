import triton 
import argparse
import torch
from torch._dynamo.backends.common import aot_autograd
from transformers import AutoConfig, AutoModelForCausalLM
from fusion.compiler import fusion_compiler


torch.set_float32_matmul_precision('high')

model_id_map = {
    "gpt": "openai-community/gpt2",
    "llama": "meta-llama/Llama-3.2-1B",
    "gemma": "google/gemma-2b",
    "qwen": "Qwen/Qwen2.5-1.5B"
}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    model_id = model_id_map[args.model]
    config = AutoConfig.from_pretrained(model_id)
    
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        model.eval()
        model_fused = torch.compile(model, 
                                    backend=aot_autograd(fw_compiler=fusion_compiler))
        input_ids = torch.randint(
            low=0, 
            high=config.vocab_size,
            size=(8, 1024)
        )

    with torch.no_grad():
        # ======= Triton Profiling ==============
        eager_ms = triton.testing.do_bench(lambda: model(input_ids))
        fused_ms = triton.testing.do_bench(lambda: model_fused(input_ids))
        
        print(
            f"\neager: {eager_ms:.3f} ms"
            f"\nfused: {fused_ms:.3f} ms"
            f"\nspeed up: {eager_ms/fused_ms:.3f}x"
        )

    

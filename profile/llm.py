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
        model_ind = torch.compile(model)
        input_ids = torch.randint(
            low=0, 
            high=config.vocab_size,
            size=(32, 1024)
        )

    with torch.no_grad():
        # ======= Triton Profiling ==============
        eager_ms = triton.testing.do_bench(lambda: model(input_ids))
        fused_ms = triton.testing.do_bench(lambda: model_fused(input_ids))
        ind_ms = triton.testing.do_bench(lambda: model_ind(input_ids))
        
        data = [(eager_ms, "eager"), (fused_ms, "fused"), (ind_ms, "inductor")]
        data = sorted(data)
        
        print("\n", "="*20 , "Results", "="*20)
        for ms, label in data:
            print(f"{label}: {ms:.3f} ms ({eager_ms/ms:.2f}x speed up)")

    

#/usr/bin/bash

nsys profile \
    -t cuda,nvtx \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --force-overwrite true \
    -o gpt \
    python llm.py --model gpt
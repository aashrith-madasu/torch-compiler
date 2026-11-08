#/usr/bin/bash

PROGRAM="mlp"

nsys profile \
    -t cuda,nvtx \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --force-overwrite true \
    -o mlp_both \
    python $PROGRAM.py
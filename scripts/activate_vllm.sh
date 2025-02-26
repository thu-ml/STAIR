vllm serve actor \
    --served-model-name actor \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --port 80 

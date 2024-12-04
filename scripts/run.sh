#!/bin/bash
source /ext3/miniconda3/bin/activate vllm

cd ../src

python main.py \
    --output_dir ../output \
    --prompt_fn ../prompts/ultimatum.yaml \
    --model gpt-4o-mini \
    --api_key HE_OPENAI_KEY
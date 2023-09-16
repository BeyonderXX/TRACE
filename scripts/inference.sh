#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port inference/infer_multi.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --dataset_name FOMC,C-STANCE \
    --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b \
    --inference_model_path /workspace/output/llama-7b-chat-seq-cl/OGD \
    --inference_batch 16 \
    --max_prompt_len 2048 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    "--CL_method","OGD" \
    --inference_output_path /workspace/output/llama-7b-chat-inference
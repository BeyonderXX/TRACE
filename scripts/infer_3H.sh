#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:4 --master_port $port inference/infer_3H.py \
   --data_path /mnt/data/user/zhang_yuansen/LLM-Continual-Learning/inference/HHH \
   --inference_tasks helpful,harmless \
   --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-13b-chat \
   --inference_batch 2 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --inference_output_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/3H > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/3H/infer_13b.log 2>&1 &
   

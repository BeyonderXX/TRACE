#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
   --data_path /mnt/data/user/zhang_yuansen/LLM-CL-Benchmark_5000 \
   --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b-chat \
   --inference_model_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/lora \
   --inference_batch 4 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method lora \
   --inference_output_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/lora/predictions > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/lora/infer.log 2>&1 &
   

#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port inference/infer_single.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b \
    --inference_model_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive \
    --inference_batch 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive/predictions > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/naive/infer.log 2>&1 &
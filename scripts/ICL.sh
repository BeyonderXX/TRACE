#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port inference/ICL.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,MeetingBank,Py150,20Minuten \
    --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b-chat \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /workspace/output/llama-7b-chat-inference-ICL
#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port inference/infer_multi.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,MeetingBank,Py150,Papyrus-f \
    --model_name_or_path /workspace/output/llama-7b-chat-seq-naive \
    --inference_batch 16 \
    --max_prompt_len 2048 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --inference_output_path /workspace/output/llama-7b-chat-inference
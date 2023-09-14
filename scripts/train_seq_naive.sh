#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port training/main.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,MeetingBank,Py150,Papyrus-f \
    --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b-chat \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --max_prompt_len 2048 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method base \
    --output_dir /workspace/output/llama-7b-chat-seq-naive-2
#!bin/bash

port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port training/replay.py  \
    --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,MeetingBank,Py150 \
    --replay_dataset_name Lima \
    --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/opt-1.3b \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --past_task_ratio 0.1 \
    --output_dir /workspace/output/llama-7b-replay


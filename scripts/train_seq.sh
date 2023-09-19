#!/bin/bash
set -x


deepspeed --include=localhost:0,1,2,3 --master_port 51420 training/main.py \
   --data_path /mnt/data/user/zhang_yuansen/LLM-CL_Benchmark \
   --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 5,3,7,5,3,5,5,7 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method lora \
   --output_dir /mnt/data/user/zhang_yuansen/outputs_LLM-CL/lora > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/lora/train.log 2>&1 &

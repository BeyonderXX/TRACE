#!bin/bash

for cl_method in "LwF"
do
    port=$(shuf -i25000-30000 -n1)
    deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
        --data_path /mnt/data/user/zhang_yuansen/LLM-CL-Benchmark_100 \
        --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
        --model_name_or_path /mnt/data/user/zhang_yuansen/PTMs/llama-2-7b-chat \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 16 \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --num_train_epochs 1,1,5,5,1,5,5,5 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --zero_stage 2 \
        --deepspeed \
        --print_loss \
        --CL_method $cl_method \
        --output_dir /workspace/output/llama-7b-chat-seq-cl/$cl_method

done
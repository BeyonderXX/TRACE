#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py \
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



# for slurm
srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/main.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat \
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
    --zero_stage 3 \
    --deepspeed \
    --print_loss \
    --CL_method lora \
    --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/lora/llama7b-chat > llama2-7b-lora.log 2>&1 &



srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/main.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
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
    --zero_stage 3 \
    --deepspeed \
    --print_loss \
    --CL_method lora \
    --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/lora/vicuna-7b > vicuna-7b-lora.log 2>&1 &

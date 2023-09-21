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
    --inference_output_path /mnt/data/user/zhang_yuansen/outputs_LLM-CL/ICL > /mnt/data/user/zhang_yuansen/outputs_LLM-CL/ICL/infer.log 2>&1 &




# for slurm, single gpu
srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51417 inference/ICL.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/ICL/llama2-7b > llama2_7b_ICL_infer.log 2>&1 &




srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51408 inference/ICL.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/ICL/vicuna-7b > vicuna-7b_ICL_infer.log 2>&1 &


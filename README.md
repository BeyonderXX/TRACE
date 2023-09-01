## Easy-SFT: Easy, Fast and Affordable SFT Training of LLMs

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)

A fast, affordable, scalable and open system framework for enabling end-to-end SFT training experience to generate high-quality LLMs at all scales.

## 🚀 What is Easy-SFT 🚀

SFT framework for LLM.

## 🙏 FrameWork 🙏

.
├── inference
│   ├── chatbot.py
│   └── infer.py			# 推理程序入口
├── LICENSE
├── README.md
├── requirements.txt
├── training
│   └── main.py			# 训练程序入口
└── utils
    ├── data
    │   ├── data_cache.py
    │   ├── data_collator.py	# 输入模型前处理
    │   ├── data_utils.py		# 数据集加载和处理
    │   └── raw_datasets.py	# 不同数据集类，用来返回数据
    ├── ds_utils.py
    ├── model
    │   └── model_utils.py
    ├── module
    │   └── lora.py
    └── utils.py

### 数据预处理

加载与修改逻辑在 **utils.data.raw_datasets.py** 中

1. 如果是在 huggingface 上有的，直接使用 datasets.load_dataset(ds_name) 加载
2. 如果是本地文件，将数据集处理成 train.json 和 eval.json两个文件，文件格式如下：

```
    [{
        "prompt": "Given my personal financial information, when can I expect to retire comfortably?",
        "answer": "xxxxxxx"
    },
    {
        "prompt": "How do I develop a high-risk investment strategy based on gambling and speculative markets?",
        "answer": "xxxxxxxx"
    }]
```


数据输入模型前的预处理在 **utils.data.data_collator.py** 中进行

train.json 的内容，在训练时，按照  Q: + prompt + A: + answer 拼接的方式生成训练样本，至于Q/A的 key 是什么样的, 自己修改 raw_datasets.py 内部的处理逻辑。

eval.json 的内容，在训练时，按照 Q: + prompt + A: + answer 拼接方式生成样本，测试困惑度。在推理阶段，按 Q: + prompt + A: 的方式生成模型输入，让模型补全内容

输入给 data_collator 的内容是一个batch samples，data_collator将其按照需求，预先处理成 tensor。我们这里假设只支持 decoder-only 模型，采用left padding，这样同时方便训练和推理的批量进行。同时padding的长度是batch中最长样本的长度，而未必是最大输入长度，加速训练/推理。


### 模型训练

运行入口是 **training/main.py**，使用deepspeed作为分布式训练框架，支持 zero 1/2/3 三个阶段，越靠后越省GPU memory，但是通信成本也越高。一般情况开 zero 2，GPU顶不住再上 zero 3。

设置了 max_prompt_len 和 max_ans_len ， 两者和为训练最大输入长度，前者作为推理最大输入长度。

模型的日志，参数保存，只在第一张卡 上进行。


为了方便debug，设置了debug参数，使用opt小模型快速debug，这里给出 debug 参数。

```
deepspeed --master_port 51419 main.py  --debug --data_path Anthropic/hh-rlhf --model_name_or_path facebook/opt-1.3b --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --max_prompt_len 512 --max_ans_len 512 --learning_rate 1e-5 --weight_decay 0. --num_train_epochs 3 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --zero_stage 2 --deepspeed --print_loss --output_dir /mnt/petrelfs/wangxiao/debug_runs > debug.log 2>&1 &
```


训练参数，去除 '--debug' 参数，修改模型名。

```
deepspeed --master_port 51419 main.py  --data_path Anthropic/hh-rlhf  --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --max_prompt_len 512 --max_ans_len 512 --learning_rate 1e-5 --weight_decay 0. --num_train_epochs 3 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --zero_stage 2 --deepspeed --print_loss --output_dir /mnt/petrelfs/wangxiao/7B_3epochs_runs > 7b.log 2>&1 &
```


### 模型推理

运行入口是 **inference/infer.py**，使用 deepspeed 的 model parallism 技术，会将模型均匀地切到不同卡上进行推理。

模型的日志，预测结果保存，只在第一张卡上进行。

同样支持debug模式

```
deepspeed --num_gpus 2 --master_port 51419 infer.py --debug --data_path Anthropic/hh-rlhf --data_split 10,0,0 --model_name_or_path facebook/opt1.3b  --max_prompt_len 512 --max_ans_len 512 --seed 1234 --deepspeed --inference_output_path /mnt/petrelfs/wangxiao/SFT/debug_predictions.csv > inference_debug.log 2>&1 &
```


正常推理指令，以llama为例

```
deepspeed --num_gpus 8 --master_port 51419 predict_sft.py --data_path Anthropic/hh-rlhf --model_name_or_path /mnt/petrelfs/wangxiao/7B_3epochs_runs --max_prompt_len 512 --max_ans_len 512  --seed 1234 --deepspeed --inference_output_path /mnt/petrelfs/wangxiao/7B_3epochs_runs/predictions.csv > inference7b.log 2>&1 &
```


具体推理时的参数在 infer.py 内部改，比如默认的推理参数为：

```
model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len, pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
```

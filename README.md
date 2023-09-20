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

训练的坑， llama2 必须要按照 bf16 的格式训练，不然loss会炸。其他模型，fp16格式训练， 改 utils.ds_utils.py 的内容:

```
"fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
 
 ->
 
 "bfloat16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
```



### 模型推理

运行入口是 **inference/infer.py**，使用 deepspeed 的 model parallism 技术，会将模型均匀地切到不同卡上进行推理。

模型的日志，预测结果保存，只在第一张卡上进行。

同样支持debug模式

```
deepspeed --num_gpus 2 --master_port 51419 infer.py --debug --data_path Anthropic/hh-rlhf --data_split 10,0,0 --model_name_or_path facebook/opt1.3b  --max_prompt_len 512 --max_ans_len 512 --seed 1234 --deepspeed --inference_task --inference_output_path /mnt/petrelfs/wangxiao/SFT/debug_predictions.csv > inference_debug.log 2>&1 &
```


正常推理指令，以llama为例

```
deepspeed --num_gpus 8 --master_port 51419 predict_sft.py --data_path Anthropic/hh-rlhf --model_name_or_path /mnt/petrelfs/wangxiao/7B_3epochs_runs --max_prompt_len 512 --max_ans_len 512  --seed 1234 --deepspeed --inference_output_path /mnt/petrelfs/wangxiao/7B_3epochs_runs/predictions.csv > inference7b.log 2>&1 &
```


具体推理时的参数在 infer.py 内部改，比如默认的推理参数为：

```
model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len, pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
```


### Baseline训练和测试

训练和测试脚本中某些参数说明：

**data_path**  数据集存放路径，需要一共包括九个数据集————8个常规训练数据集（C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten）和一个Replay数据集（Lima）  （需要改）

**model_name_or_path** 模型路径（需要改）

**dataset_name**  训练所需要的数据集名称，一般是上面的八个，用”,“隔开

**num_train_epochs**  每个数据集训练所需要的epoch数量，用“,”隔开

**past_task_ratio** Replay的参数，replay之前任务多少比例的数据

**output_dir** 输出文件夹，训练完成后，每完成训练一个dataset后保存一次模型，在output_dir下会产生8个文件夹

**CL_method**  持续学习方法，全参数SFT对应“base”，lora对应“lora”，replay不用加

**inference_model_path** 训练完后模型保存的文件夹，和上面的output_dir对应，程序会遍历文件夹中的模型进行时序测试


## 时序训练和测试（naive，全参数SFT）

```
bash scripts/train_seq_naive.sh
bash scripts/infer_seq.sh
```

## Lora训练和测试

```
bash scripts/train_lora.sh
bash scripts/infer_lora.sh
```

## Replay训练和测试

测试脚本和时序训练一样

```
bash scripts/train_replay.sh
bash scripts/infer_seq.sh
```

## ICL测试

```
bash scripts/ICL.sh
```
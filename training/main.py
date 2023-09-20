#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from params import Method2Class, AllDatasetName


# TODO, check support for OPT and llama


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    # set batch size
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout
                            )
    
    # some CL methods can be realized by peft
    if args.CL_method == "LFPT5":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        initial_prompt = getInitialPrompt(tokenizer, prompt_token_number=300)
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=300,
            prompt_tuning_init_text=initial_prompt,
            tokenizer_name_or_path=args.model_name_or_path,
        )
        model = get_peft_model(model, peft_config)

    if args.CL_method == "O-LoRA":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
                
    if args.CL_method == "OGD":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True

    if args.CL_method == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    for dataset in Datasets:
        dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed
        )

        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)

        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)

        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size)
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def get_optimizer(model):
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
    
    if args.CL_method=="PP" or args.CL_method=="L2P":
        if "opt" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.decoder.embed_tokens.weight.shape
            embed_tokens = model.model.decoder.embed_tokens
            
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        elif "llama" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
            
        if args.CL_method=="PP":
            args.prefix_len = 20
            args.task_length = len(train_task_list)
            model = convert_PP_model(model, args)
            
        elif args.CL_method=="L2P":
            args.pool_size = 10
            args.prompt_length = 5
            args.prompt_init = "uniform"
            model = convert_L2P_model(model, args)
            for name, params in model.named_parameters():
                if "prompt" not in name:
                    params.requires_grad=False
                    
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        CL_Trainer.train_continual()


if __name__ == "__main__":
    main()

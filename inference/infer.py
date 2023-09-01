"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed


from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model


# dist.init_process_group(backend='nccl')



def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
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
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    # TODO, add other inference params


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
  
    # added by wangxiao
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug mode, which will use a small model and small dataset')
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # 自动获取 word_size
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()


    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    # torch.distributed.barrier()

    if args.debug:
        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
        if "falcon" in args.model_name_or_path.lower():
            tokenizer.bos_token = tokenizer.eos_token
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                                   fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    # default the LLM is decoder only model, so padding side is left
    tokenizer.padding_side = 'left'

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            debug=args.debug)

    # reference
    # https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/inference-tutorial.md
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py
    # https://discuss.huggingface.co/t/using-text-generation-pipeline-for-llama-2-7b-chat-hf-setting-high-t-doesnt-change-output/48982
    # https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/
    # https://www.deepspeed.ai/tutorials/inference-tutorial/
    
    replace_with_kernel_inject = False if "falcon" in args.model_name_or_path.lower() else True
    ds_engine = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.half, checkpoint=None,          replace_with_kernel_inject=replace_with_kernel_inject)
    model = ds_engine.module
    
    # Prepare the data
    _, infer_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_output_path,
        args.seed
    )

    inf_data_collator  = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=True
    )

    infer_sampler = SequentialSampler(infer_dataset)
    infer_dataloader = DataLoader(infer_dataset,
                                  collate_fn=inf_data_collator,
                                  sampler=infer_sampler,
                                  batch_size=args.inference_batch)

    progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(args.global_rank != 0))
    
    """
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    """
    def prediction(model, infer_dataloader):
        predicted_sequences = []
        sources_sequences = []
        model.eval()

        for step, batch in enumerate(infer_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            del batch['sources']
            batch = to_device(batch, device)
            progress_bar.update(1)
            prompt_len = batch['input_ids'].shape[1]

            # update progress bar
            if args.global_rank == 0:
                progress_bar.update(1)
                description = f"Step {step}"
                progress_bar.set_description(description, refresh=False)

            with torch.no_grad():
                # TODO, add more inference params
                generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len, 
                                              pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )

            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            predicted_sequences += sequences
            
            # if step > 20:
            #     break

        return sources_sequences, predicted_sequences

    
    def save_inference_results(sources_sequences, predicted_sequences):
        prompts = []
        results = []

        for source, predicted in zip(sources_sequences, predicted_sequences):
            prompts.append(source)
            results.append(predicted)

        # save prompts and results in a csv file
        df = pd.DataFrame({'prompts': prompts, 'results': results})
        df.to_csv(args.inference_output_path, index=False)
        print("***** Save inference results *****")
        print("Sucessful save predictions to {}".format(args.inference_output_path))

    # Inference !
    print_rank_0("***** Start inference *****", args.global_rank)
    sources_sequences, predicted_sequences = prediction(model, infer_dataloader)


    if args.global_rank <= 0:
        print("***** Start inference results *****")
        save_inference_results(sources_sequences, predicted_sequences)


if __name__ == "__main__":
    main()

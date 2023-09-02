import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from copy import deepcopy
from transformers import AdamW

from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer



class PP_test:
    def __init__(self,
                 model, 
                 tokenizer,
                 args
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        
    def train(self, task_list):
        train_dataloader = task_list['task1']
        device='cuda'
        total_steps = self.args.num_train_epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(self.args.num_train_epochs):

            self.model.train()

            for step, batch in enumerate(train_dataloader):
                batch = to_device(batch, device)
                batch = {k:batch[k].to('cuda') for k in batch}
                lm_labels = batch["labels"]
                # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

                # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
                inputs_embeds = self.model.module.model.decoder.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】

                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'], labels=lm_labels,use_cache=False)
                loss = outputs.loss

                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
        
            
                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
import os
import time
import torch
import pickle
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


def getInitialPrompt(tokenizer, prompt_token_number, task_name):
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-large") 
    fr = open('model/Replay/allnumber.pickle', 'rb') # t5词频表
    all_tokens = pickle.load(fr)
    sorted_all_tokens = sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)
    top5000_t5 = []
    for one in sorted_all_tokens:
        if len(top5000_t5) < 5000:
            top5000_t5.append(t5_tokenizer.decode(one[0]))

    # 以t5词表为基准，如果当前词表中有对应的token则加入，否则随机选择
    top5000 = []
    vocab = []
    for token_id in range(len(tokenizer)):
        vocab.append(tokenizer.decode(token_id))
    for token in top5000_t5:
        if token in vocab:
            top5000.append(token)
        else: 
            random_token = random.choice(vocab)
            top5000.append(random_token)

    # 从top5000里随机挑选token直至prompt的最大长度
    initial_prompt = task_name + " "
    tokens_to_use = random.sample(top5000, prompt_token_number)
    initial_prompt += " ".join(tokens_to_use)
    input_ids = tokenizer.encode(initial_prompt)[:prompt_token_number]
    initial_prompt = tokenizer.decode(input_ids)

    return initial_prompt


class LFPT5(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, args,
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, args)
        '''
        1. prompt tuning
        2. pseudo data generation
        '''
        # 增添2种特殊字符：
        # 1. "__ans__"                  伪数据prompt/answer的分隔符
        # 2. "__" + task name + "__"    生成伪数据任务的起始符
        tasks_name = self.args.dataset_name
        self.tokenizer.add_tokens("__ans__")
        for task_name in tasks_name:
            self.tokenizer.add_tokens("__" + task_name + "__")

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)


    def evaluation(self, eval_dataloader):
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
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


    def train_one_task(self, task, i_task, epochs):
        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # TODO: add loss_lm

                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

            # Evaluate perplexity on the validation set.
            print_rank_0(
                f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
                self.args.global_rank)
            perplexity = self.evaluation(eval_dataloader)
            print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            self.model.tput_timer.update_epoch_count()

        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            try:
                if not os.path.exists(peft_model_id):
                    os.makedirs(peft_model_id)
            except:
                None
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)
            
            if i_task < num_task - 1:
                print_rank_0(f'Let\'s have a little break to get ready for the next task! ^-ω-^Zzz...', self.args.global_rank)
                time.sleep(10)
            else:
                print_rank_0(f'Mission Complete! \^·ω·^/', self.args.global_rank)


    def save_model(self):
        pass

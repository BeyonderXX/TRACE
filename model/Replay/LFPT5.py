import os
import time
import torch
import pickle
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def getInitialPrompt(tokenizer, prompt_token_number):
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
    tokens_to_use = random.sample(top5000, prompt_token_number)
    initial_prompt = " ".join(tokens_to_use)
    input_ids = tokenizer.encode(initial_prompt)[:prompt_token_number]
    initial_prompt = tokenizer.decode(input_ids)

    return initial_prompt


class LFPT5(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, args,
                 lambda_lm = 0.1
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

        self.lambda_lm = lambda_lm # LM损失的权重系数


    def get_dataloader(self, task, pseudo_prompt=None, pseudo_answer=None, isLM=False):
        '''
        generate datasets for pseudo data generation
        '''
        # get raw data
        train_dataset, _, _ = create_prompt_dataset(
            self.args.local_rank,
            os.path.join(self.args.data_path,task),
            self.args.data_output_path,
            self.args.seed
        )
        
        if isLM:
            # create a dataset for training pseudo-data-genertation ability
            # 'lm' stands for language modeling
            prompt_dataset = train_dataset.prompt_dataset
            answer_dataset = train_dataset.answer_dataset
            prompt_dataset_lm = []
            answer_dataset_lm = []
            for idx in range(len(prompt_dataset)):
                prompt_dataset_lm.append("__" + task + "__")
                answer_dataset_lm.append(prompt_dataset[idx] + "__ans__" + answer_dataset[idx])
            train_dataset.prompt_dataset = prompt_dataset_lm
            train_dataset.answer_dataset = answer_dataset_lm
        
        if pseudo_prompt != None or pseudo_answer != None:
            # add pseudo data
            train_dataset.prompt_dataset += pseudo_prompt
            train_dataset.answer_dataset += pseudo_answer

        # create dataLoaders
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        data_collator  = DataCollator(
            self.tokenizer,
            padding="longest",
            max_prompt_len=self.args.max_prompt_len,
            max_ans_len=self.args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        train_dataloader = DataLoader(train_dataset,
                            collate_fn=data_collator,
                            sampler=train_sampler,
                            batch_size=self.args.per_device_train_batch_size)
        return train_dataloader


    def generate_pseudo_data(self, i_task, times_of_generation=100):
        i = 0
        for task_name in self.train_task_list:
            if i == i_task:
                print_rank_0(f"pseudo data generation is completed", self.args.global_rank)
                break 
            # generate pseudo data of the previous task, given a special token __<task_name>__
            print_rank_0(f"generating pseudo data for the task: " + task_name, self.args.local_rank)
            self.model.eval()
            input_ids = self.tokenizer.encode("__" + task_name + "__", return_tensors='pt').to(self.device)
            input_ids = input_ids.repeat(self.args.per_device_eval_batch_size,1)
            attention_mask = torch.ones_like(input_ids)
            max_len = self.args.max_prompt_len + self.args.max_ans_len
            pseudo_prompt = []
            pseudo_answer = []
            pseudo_prompt_lm = []
            pseudo_answer_lm = []
            for time in range(times_of_generation):
                print_rank_0(f"Generating pseudo data. " + str(times_of_generation - time) + " more times left.", self.args.local_rank)
                output = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=max_len,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.5,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=5,
                    )
                generated_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                for generated_text in generated_texts:
                    if "__ans__" in generated_text:
                        generated_text.replace("__" + task_name + "__", "")
                        pseudo_prompt.append(generated_text.split("__ans__")[0])
                        pseudo_answer.append(generated_text.split("__ans__")[1])
                        pseudo_prompt_lm.append("__" + task_name + "__")
                        pseudo_answer_lm.append(generated_text)
            print_rank_0(f"number of available pseudo prompts:" + str(len(pseudo_prompt)), self.args.local_rank)
            i += 1
        return pseudo_prompt, pseudo_answer, pseudo_prompt_lm, pseudo_answer_lm
        

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
        eval_dataloader = self.eval_task_list[task]
        if i_task > 0:
            pseudo_prompt, pseudo_answer, pseudo_prompt_lm, pseudo_answer_lm = self.generate_pseudo_data(i_task)
            print_rank_0(f"adding pseudo data to the original dataset...", self.args.global_rank)
            train_dataloader = self.get_dataloader(task, pseudo_prompt, pseudo_answer, False)
            train_dataloader_lm = self.get_dataloader(task, pseudo_prompt_lm, pseudo_answer_lm, True)
        else: 
            train_dataloader = self.train_task_list[task]
            train_dataloader_lm = self.get_dataloader(task, None, None, True)

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, (batch, batch_lm) in enumerate(zip(train_dataloader, train_dataloader_lm)):
                del batch['sources']
                del batch_lm['sources']
                batch = to_device(batch, self.device)
                batch_lm = to_device(batch_lm, self.device)
                outputs = self.model(**batch, use_cache=False)
                outputs_lm = self.model(**batch_lm, use_cache=False)
                loss = outputs.loss + outputs_lm.loss * self.lambda_lm
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

import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class O_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1 = 0.5, lamda_2 = 0
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        orthological to previous adapters
        '''
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        
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
                ########################### Regularization ##########################
                orthogonal_loss = 0.
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        for name_, param_ in self.model.named_parameters():
                            if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                                break # target modules have been matched

                # l2-normalization for loranew_A/B
                l2_loss = 0.
                for name, param in self.model.named_parameters():
                    if "loranew_" in name:
                        l2_loss += torch.norm(param, p=2)

                print_rank_0(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {self.lamda_1}; λ2: {self.lamda_2}", self.args.global_rank)
                loss = loss + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                ######################################################################
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

        #### COMBINE lora with lora_new and INITIALIZE lora_new ####
        state_dict = self.model.state_dict()    
        for k in state_dict:
            if "lora_A" in k:
                for k_ in state_dict:
                    if "loranew_A" in k_ and k.split("lora_A")[0] == k_.split("loranew_A")[0]:
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, r]
                        break # target modules have been matched
            elif "lora_B" in k:
                for k_ in state_dict:
                    if "loranew_B" in k_ and k.split("lora_B")[0] == k_.split("loranew_B")[0]:
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [r, r_sum + r]
                        break # target modules have been matched
            elif "loranew_A" in k:
                nn.init.kaiming_uniform_(state_dict[k], a=math.sqrt(5))
            elif "loranew_B" in k:
                nn.init.zeros_(state_dict[k])
        
        #### RESET ####
        for name, param in self.model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False

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

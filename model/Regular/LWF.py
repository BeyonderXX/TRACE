import numpy as np
import torch
import quadprog
import random
from tqdm.auto import tqdm
import copy
import json
import torch.nn.functional as F

class LwF(object):
    def __init__(self,
                 model,
                 task_list,
                 args,
                 ):
        self.model = model
        self.task_list = task_list
        self.args = args
            
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'])
        loss = outputs[0]

        return loss
    
    
    def KD_loss(self, new_logits, prev_logits, T):
        prev_logits = prev_logits.to('cuda')
        prev_logits = F.log_softmax(prev_logits/T, dim=1)
        new_logits = F.softmax(new_logits/T, dim=1)
        kd_loss = torch.sum(prev_logits * new_logits, dim=1, keepdim=False)
        kd_loss = -torch.mean(kd_loss, dim=0, keepdim=False)
        
        
        return kd_loss
    
    def new_input_old_model_logits(self, i_task):
        train_dataloader = self.task_list[i_task+1]
        new_task_logits = {}
        for step, batch in enumerate(tqdm(train_dataloader)):
            del batch['sources']
            batch = {k:batch[k].to('cuda') for k in batch}
            outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            new_task_logits[str(step)] = logits
            
        new_task_logits = json.dumps(new_task_logits)
        with open(self.args.output_dir+"/LwF/{}.json".format(i_task+1),"w") as w:
            w.write(new_task_logits)
            

        
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40
                       ):
        if i_task!=0:
            with open(self.args.output_dir+"/LwF/{}.json") as f:
                prev_logits = json.load(f)

        dataloader_train = self.task_list[task]
        for epoch in range(epochs):
            print(epoch)
            self.model.train()
            total_steps = self.args.num_train_epochs * len(dataloader_train)
            progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    
                if i_task!=0:
                    loss += self.KD_loss(loss.logits, prev_logits[str(step)], 2)
                
                self.model.backward(loss)
                self.model.step()
                
        if i_task+1 < len(self.task_list):
            self.new_input_old_model_logits(i_task)



    
    # Train model continually
    def train_continual(self,
                        epochs=40,
                        ):
        self.observed_tasks.append(0)
        for num, task in enumerate(self.task_list):
            self.train_one_task(task, num, epochs)
            

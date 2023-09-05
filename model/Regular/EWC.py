from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model


class EWC(CL_Base_Model):
    def __init__(self):
        super().__init__()
        self.device="cuda"
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._previous_params = {}

        for n, p in deepcopy(self.params).items():
            self._previous_params[n] = p.data # Previous task parameters
        self.grads = {} # 存储节点名称与节点的grad
        
        self.fisher = {}
        self.init_fisher()


    
    def init_fisher(self):
        for n, p in deepcopy(self.params).items():
            p.data.zero_()  #所有参数置零
            self.fisher[n] = p.data  #初始化零矩阵
            
    #计算每个参数的Fisher信息矩阵的值：每个样本输入模型，每个参数计算梯度的平方和，除以总的样本数量
    def _update_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data += self.grads[n].data ** 2
    #正则化，除以训练集长度
    def _regular_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data /= self.train_length

    
    def _update_previous_params(self):
        for n, p in self.model.named_parameters():
            self._previous_params[n] = p.data # Previous task parameters

    #计算惩罚loss
    def penalty(self):
        restrict_loss = 0
        precision_matrices = self._diag_fisher()
        for n, p in self.model.named_parameters():
            restrict_loss_params = precision_matrices[n] * (p - self._previous_params[n]) ** 2
            restrict_loss += restrict_loss_params.sum()
        return restrict_loss
    
    def train_step(self,
                    batch):

        # batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        # inputs_embeds = self.model.model.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】

        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'])
        
        loss = outputs[0]
        if self.task_num!=0:
            restrict_loss = self.penalty()
            loss += restrict_loss

        return loss
    
    def save_grad(self,name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0)
            grad = torch.clamp(grad, -self.args.ds_config['gradient_clipping'], self.args.ds_config['gradient_clipping'])
            self.grads[name] = grad
        return hook
    def retain_grad(self):
        for n,p in self.model.named_parameters():
            p.register_hook(self.save_grad(n))
    
    
    def train_one_task(self,
                       task,
                       epochs=40):

        print('task = ', task)

        dataloader_train = self.task_list[task]
        self.train_length = len(dataloader_train)
        total_steps = self.args.num_train_epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))


        for epoch in range(epochs):
            print(epoch)
            self.model.train()

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                self.model.backward(loss)
                self.model.step()
                self._update_fisher()


    
    # Train model continually
    def train_continual(self):
        #在训练之前确定梯度
        self.retain_grad()

        for num, task in enumerate(self.task_list):
            self.task_num=num
            self.train_one_task(task, self.args.num_train_epochs)
            self._regular_fisher()  
            
            '''
            每一步之后更新？还是一个任务更新一次？
            '''
            self._update_previous_params()


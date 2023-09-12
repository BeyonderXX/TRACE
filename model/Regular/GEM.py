import numpy as np
import torch
import quadprog
import random
from tqdm.auto import tqdm
from deepspeed.utils import safe_get_full_grad
from model.base_model import CL_Base_Model


class GEM(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.observed_tasks = []
        self.grad_dims = [] #存储每层的参数数量
        for name, param in self.model.named_parameters():
            self.grad_dims.append(param.data.numel())
        self.n_tasks = len(self.train_task_list.keys())
        self.grads = torch.zeros([sum(self.grad_dims), self.n_tasks], dtype=torch.bfloat16).cuda()  #存储每个任务的梯度
        self.cnt=len(self.grad_dims)


                
    def store_grads(self, grads, grad_dims, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        # grads[:, tid].fill_(0.0)
        cnt = 0
        for name, param in self.model.named_parameters():
            hp_grad = safe_get_full_grad(param)
            if hp_grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en, tid].copy_(torch.nan_to_num(hp_grad.data.view(-1),nan=0))
            cnt += 1
                
    def project2cone2(self,gradient,indx, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            反向传播,梯度从后往前传,memory从后往前匹配
            
        """
        raw_shape = gradient.shape
        gradient = gradient.view(-1)
        beg = sum(self.grad_dims[:self.cnt-1])
        end = sum(self.grad_dims[:self.cnt])
        self.cnt-=1
        # print(self.cnt)

        memories = self.grads[beg:end].index_select(1,indx)
        # dotp = torch.mm(self.grads[:, i_task].unsqueeze(0),
        #                         self.grads.index_select(1, indx))
        dotp = torch.mm(gradient.unsqueeze(0), memories)
        
        gradient = gradient.unsqueeze(1)

        
        if (dotp < 0).sum() != 0:
            memories_np = memories.cpu().t().double().numpy()
            gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
            t = memories_np.shape[0]
            P = np.dot(memories_np, memories_np.transpose())
            P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
            q = np.dot(memories_np, gradient_np) * -1
            G = np.eye(t)
            h = np.zeros(t) + margin
            v = quadprog.solve_qp(P, q, G, h)[0]
            x = np.dot(v, memories_np) + gradient_np
            gradient.copy_(torch.Tensor(x).view(-1, 1))
            
        #保证输入的梯度和输出的梯度shape一样
        return gradient.view(raw_shape)
            
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'])
        loss = outputs[0]

        return loss
    
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40
                       ):
        
        '''
        replay?
        '''
        self.observed_tasks.append(i_task)
        # for _ in range(self.a_gem_n):
        #     rand_task = random.choice(range(i_task))

        #     self.forward_pass(self.model, rand_task)

        # store_grads(pp=self.model.named_parameters(),
        #                         grads=self.grads,
        #                         grad_dims=self.grad_dims,
        #                         tid=i_task).cuda()

        dataloader_train = self.train_task_list[task]
        indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) 
        if indx.shape[0]!=0:
            for name, params in self.model.named_parameters():
                
                params.register_hook(lambda grads: self.project2cone2(
                                                                        grads,
                                                                        indx))
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
                
                #设置梯度更新的方法
                self.model.backward(loss)
                # for name, param in self.model.named_parameters():
                #     hp_grad = safe_get_full_grad(param)
                #     if self.args.global_rank<=0:
                #         print("{}:{}".format(name,hp_grad))
                
                #恢复滑动指针的值
                self.cnt = len(self.grad_dims)
                self.store_grads(
                        grads=self.grads,
                        grad_dims=self.grad_dims,
                        tid=i_task)
                    
                self.model.step()



            

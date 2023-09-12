from copy import deepcopy

import torch
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from deepspeed.utils import safe_get_full_grad


def convert_L2P_model(model, args):
    def init_new_prompt(pool_size,prompt_len):
        N = args.embed_tokens_length
        prompt_weigths = []
        for t in range(pool_size):
            prompt_weight = []
            for i in range(prompt_len):
                with torch.no_grad():
                    j = np.random.randint(N) # random token
                    w = deepcopy(args.embed_tokens.weight[j].detach().cpu().numpy())
                    prompt_weight.append(w/100)
                    # prompt_weigths.append(w)
            prompt_weigths.append(prompt_weight)

        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths
    # prompt_pool_shape = (args.pool_size, args.prompt_length, args.embed_tokens_dim)
    # if args.prompt_init == 'zero':
    #     model.model.prompt = nn.Parameter(torch.zeros(prompt_pool_shape, dtype=torch.bfloat16,requires_grad=True)).to("cuda")
    if args.prompt_init == 'uniform':
        model.model.prompt = nn.Parameter(torch.tensor(init_new_prompt(args.pool_size , args.prompt_length),requires_grad=True))
            
    return model



class L2P(CL_Base_Model):
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 embedding_key='mean',
                 top_k=3,
                 batchwise_prompt=False,
                 pull_constraint_coeff=0.5
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        
        self.embed_dim = self.args.embed_tokens_dim
        self.embed_tokens = self.args.embed_tokens
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.pull_constraint_coeff = pull_constraint_coeff
        
        # use mean of prompt as key
        # only compatible with prompt, not prefix
        prompt_mean = torch.mean(self.model.model.prompt, dim=1)
        self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def train_step(self, batch, prompt_mask=None, cls_features=None):
        input_ids = batch['input_ids']
        attn_masks = batch['attention_mask']
        labels = batch['labels']
        inputs_embeds = self.embed_tokens(input_ids)
        
        out = dict()
        if self.embedding_key == 'mean':
            inputs_embeds_mean = torch.mean(inputs_embeds, dim=1)
        elif self.embedding_key == 'max':
            inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0] + 2 * torch.mean(inputs_embeds, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0] # B, C
            else:
                inputs_embeds_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to("cuda") # Pool_size, C
        inputs_embeds_norm = self.l2_normalize(inputs_embeds_mean, dim=1) # B, C

        similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size
        
        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(inputs_embeds.shape[0], -1) # B, top_k
        else:
            idx = prompt_mask # B, top_k

        batched_prompt_raw = self.model.model.prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
        inputs_embeds = torch.cat([batched_prompt, inputs_embeds],axis=1)
        
        prefix_length = batched_prompt.shape[1]
        attn_masks = torch.concat((torch.tensor(1).to("cuda").repeat(batch_size,prefix_length),attn_masks), axis=1)
        labels = torch.concat((labels[0][0].repeat(batch_size,inputs_embeds.shape[1]-labels.shape[1]),labels),axis=1)
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn_masks)
        loss = outputs[0]
        
        batched_key_norm = prompt_norm[idx]
        inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        loss -= reduce_sim * self.pull_constraint_coeff
        return loss


        
        

        # out['prompt_idx'] = idx

        # # Debugging, return sim as well
        # out['prompt_norm'] = prompt_norm
        # out['inputs_embeds_norm'] = inputs_embeds_norm
        # out['similarity'] = similarity

        # # Put pull_constraint loss calculation inside
        # batched_key_norm = prompt_norm[idx] # B, top_k, C
        # out['selected_key'] = batched_key_norm
        # inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        # sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        # reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        # out['reduce_sim'] = reduce_sim

        
        # # The input with the prompt concatenated to the front. [B, prompt+token, C]
        # out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, inputs_embeds], dim=1)
    
    
    
    
    def train_one_task(self, task, i_task, epochs):
        print('task = ', task)

        dataloader_train = self.train_task_list[task]
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
                self.model.backward(loss, retain_graph=True)
                # for n, lp in self.model.named_parameters():
                #     # 1. gradient lookup
                #     # For zero1 and zero2, gradient lookup must be called after `backward` and before `step`
                #     # For zero3, gradient lookup must be called after `backward`
                #     hp_grad = safe_get_full_grad(lp)
                #     if self.args.global_rank == 0:

                #         print(hp_grad)
                self.model.step()
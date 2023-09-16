from copy import deepcopy

import torch
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from deepspeed.utils import safe_get_full_grad
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
import json
import os
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


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
                    prompt_weight.append(w)
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
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn_masks, use_cache=False)
        loss = outputs[0]
        
        batched_key_norm = prompt_norm[idx]
        inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        loss -= reduce_sim * self.pull_constraint_coeff
        return loss
    
    
    
    
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
                #     if "prompt" in n:
                #         hp_grad = safe_get_full_grad(lp)
                #         if self.args.global_rank == 0:

                #             print(hp_grad)
                self.model.step()
                
                
    def evaluate(self, round, infer_task_id, task):
        self.evaluate_one_task(round,infer_task_id, task)
        
    def evaluate_one_task(self, round, infer_task_id, task):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        infer_dataloader = self.test_task_list[task]

        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            label_sequences = []
            model.eval()

            for step, batch in enumerate(infer_dataloader):
                ground_truths_ids = self.tokenizer(batch['gts'], 
                                                   truncation=True,
                                                   max_length=self.args.max_ans_len,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   return_tensors='pt')['input_ids'].to(device)
                del batch['gts']
                del batch['sources']
                batch = to_device(batch, device)
                progress_bar.update(1)

                # update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Step {step}"
                    progress_bar.set_description(description, refresh=False)

                with torch.no_grad():
                    # sft config
                    input_ids = batch['input_ids']
                    attn_masks = batch['attention_mask']
                    inputs_embeds = self.embed_tokens(input_ids)

                    if self.embedding_key == 'mean':
                        inputs_embeds_mean = torch.mean(inputs_embeds, dim=1)

                    prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to("cuda") # Pool_size, C
                    inputs_embeds_norm = self.l2_normalize(inputs_embeds_mean, dim=1) # B, C

                    similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size

                    _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                        

                    batched_prompt_raw = self.model.model.prompt[idx] # B, top_k, length, C
                    batch_size, top_k, length, c = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
                    inputs_embeds = torch.cat([batched_prompt, inputs_embeds],axis=1)
                    
                    prefix_length = batched_prompt.shape[1]
                    attn_masks = torch.concat((torch.tensor(1).to("cuda").repeat(batch_size,prefix_length),attn_masks), axis=1)                    
                    
                    generate_ids = model.generate(inputs_embeds=inputs_embeds,
                                                  attention_mask=attn_masks,
                                                  max_new_tokens=self.args.max_ans_len,
                                                  bos_token_id=self.tokenizer.bos_token_id,
                                                  eos_token_id=self.tokenizer.eos_token_id,
                                                  pad_token_id=self.tokenizer.unk_token_id,
                                                  generation_config=generation_config,
                                                use_cache=False

                                                  )
                # add for distributed 
                gathered_ids, max_seq_len = self.dist_results_gather(generate_ids, self.tokenizer.eos_token_id)
                gathered_labels, max_label_len = self.dist_results_gather(ground_truths_ids, self.tokenizer.eos_token_id)

                if self.args.global_rank <= 0:
                    sou_sequences = self.tokenizer.batch_decode(gathered_ids[:, : max_seq_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pre_sequences = self.tokenizer.batch_decode(gathered_ids[:, max_seq_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    lab_sequences = self.tokenizer.batch_decode(gathered_labels[:, : max_label_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    predicted_sequences += pre_sequences
                    sources_sequences += sou_sequences
                    label_sequences += lab_sequences

            return sources_sequences, predicted_sequences, label_sequences


        def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                   ground_truths: list, round: int, i_task: int, task: str):
            # save as a json file
            df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                  'labels': ground_truths}
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)

            with open(self.args.output_dir + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
                json.dump(df, file, ensure_ascii=False)


        # Inference !
        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(self.model, infer_dataloader)

        # Get Accuracy/ROUGE/BLEU/...
        # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
        if self.args.global_rank <= 0:
            if task == "ScienceQA":
                evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
            elif task == "MeetingBank":
                evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
            elif task == "C-STANCE":
                evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
            elif task == "Papyrus-f":
                evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
            elif task == "Py150":
                evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
            elif task == "FOMC":
                evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
            elif task == "NumGLUE-cm":
                evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
            elif task == "NumGLUE-ds":
                evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
            # elif task == "ToolBench":
            #     evaluation_result = eval_ToolBench.eval(predicted_sequences, ground_truths)
            else:
                evaluation_result = {}

            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, infer_task_id, task)
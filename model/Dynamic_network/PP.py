import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from copy import deepcopy
from transformers import AdamW
from model.base_model import CL_Base_Model


class ResMLP(torch.nn.Module):
    def __init__(self, 
                 bottleneck_size,
                 module_type='MLP2',
                 emb_dimension=512,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer'). 
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        if module_type=='MLP1':
            # if layer_norm:
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size, dtype=torch.bfloat16),
                nn.ReLU(),
                nn.Linear(bottleneck_size, emb_dimension, dtype=torch.bfloat16),
                nn.LayerNorm(emb_dimension, dtype=torch.bfloat16),
            )
            # else:
            #     self.module = nn.Sequential(
            #         nn.Linear(emb_dimension, bottleneck_size),
            #         nn.Tanh(),
            #         nn.Linear(bottleneck_size, emb_dimension),
            #     )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)



class PP(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 prefix_len=20,
                 prefix_path=None, # path to the pre-trained progressive prompt
                 freeze_weights=True,
                 seq_len=512,
                 early_stopping=True,
                 prefix_MLP="MLP2",
                 bottleneck_size=800, # bottleneck size in case of using MLP reparametrization
                 ):
        
        """Class for CL & prompt tuning experiments with T5 model.
        Args:
            model_name (str): T5 model type to use (e.g. base/small/large etc.)
            task_list (List[str]): list of downstream tasks to be trained on. In case of 1 task - regular training.
            batch_size (int, optional): Batch size used. Defaults to 8.
            select_k_per_class (int, optional): Limit data to k samples/class. Defaults to -1 (keep original dataset size).
            prefix_len (int, optional): Prompt length to use. Defaults to 0 (i.e. no prompt).
            prefix_path (str, optional): Path to the pre-trained progressive prompt. Defaults to None.
            freeze_weights (bool, optional): Whether to freeze model weights. Defaults to True (prompt tuning setup).
            freeze_except (str, optional): Freeze all weights except parameters matching this condition. 
                Defaults to 'shared' (freeze all weights except word embeddings).
            lr (float, optional): Learning rate. Defaults to 0.3.
            weight_decay (float, optional): Weight decay coefficient. Defaults to 1e-5.
            seq_len (int, optional): Input text lengths in tokens. Defaults to 512.
            early_stopping (bool, optional): Use early stopping to select best prompt/model. Defaults to True.
            prefix_MLP (str, optional): what MLP to use for prompt re-parameterization. Defaults to 'MLP-1'.
            bottleneck_size (int, optional): Bottleneck size in case of using MLP reparametrization. Defaults to 800.
            mlp_lr (float, optional): MLP learning rate to use. Defaults to None (lr value will be used).
            weight_decay_mlp (float, optional): Wight decay coefficient in MLP. Defaults to None.
            get_test_subset (bool, optional): Whether to create a test subset. Defaults to True.
            memory_perc (float, optional): Percentage of data saved for memory replay in CL settings. Defaults to 0.0.
                 
                 
            prefix_len (int, optional): Soft prompt length (only needed if virtual tokens are added to the vocab). Defaults to 0.
            freeze_weights (bool, optional): Whether to freeze base model weights. 
                Model weights need to be frozen for prompt tuning (i.e. True)! Defaults to False.
            freeze_except (str, optional): If freeze_weights, do not freeze weights that contain this text. 
                Defaults to 'shared' (will avoid freezing word embeddings layer in T5).
            lr (float, optional): Prompt (model) learning rate. Defaults to 0.1.
            weight_decay (float, optional): Prompt (model) weight decay coefficient. Defaults to 0.00.
            prompt_name (str, optional): Shared name for prompt virtual tokens (when added to the vocab). 
                Not used in the latest implementation. Defaults to 'PRE'.
            
            prefix_MLP (str, optional): . Defaults to 'None'.
            mlp_bottleneck (int, optional): MLP bottleneck dimension. Defaults to 1000.
            weight_decay_mlp (float, optional): MLP weight decay coefficient. Defaults to 0.01.
            mlp_lr (float, optional): MLP learning rate. Defaults to 1e-4.
            mlp_layer_norm (bool, optional): Whether to use LN in MLP. Defaults to True.
            
            early_stopping (bool, optional): Whether to select best paramteres via early stopping. Defaults to True.
            opt (str, optional): Optimizer to use. Curretnly AdamW and LAMB are supported. Defaults to 'AdamW'.
        
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs ('MLP1' and 'MLP2'). Defaults to 'MLP1'.
            emb_dimension (int, optional): . Defaults to 512.
            layer_norm (bool, optional): . Defaults to True.
        
        
        
        """
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.freeze_weights = freeze_weights
        self.seq_len = seq_len
        self.early_stopping = early_stopping
        self.embed_tokens_dim = self.args.embed_tokens_dim
        self.embed_tokens_length = self.args.embed_tokens_length
        self.embed_tokens = self.args.embed_tokens
        self.prefix_MLP = prefix_MLP


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Freezing model weights for prompt tuning
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights()
           
        self.prefix_len = prefix_len
        # Creating a trainable soft prompt
        if prefix_len>0:
            if prefix_path==None:
                self.previous_prompts = torch.zeros([0, self.model.model.prompt.shape[1]],
                                                    requires_grad=False, dtype=torch.bfloat16).to(self.device)
            else: # initializing previous prompts from the path
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(np.load(prefix_path), requires_grad = False).to(self.device)
    

    # Concatenate newly learned prompt to the joint "Progressive Prompts"
    def progress_previous_prompts(self, task_num=None):


        if task_num!=None and self.prefix_MLP!=None:
            new_prompt = self.model.model.mlps[task_num](self.model.model.prompt)
        else:
            new_prompt = self.model.model.prompt
            # new_prompt = new_prompt.detach().cpu().numpy()

        # new_prompt = torch.tensor(new_prompt, requires_grad = False).to(self.device)
        self.previous_prompts = torch.concat([new_prompt, self.previous_prompts], axis=0)
        print('Updated progressive prompts ', self.previous_prompts.shape)
    

    # Perform one train step for prompt tuning (following Lester et al.)
    def train_step_lester(self,
                          batch,
                          task=None,
                          task_num=None,
                          progressive=True):
        embed_prompt = self.prefix_MLP!=None
        if embed_prompt:
            assert task!=None
            mlp = self.model.model.mlps[task_num]

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        inputs_embeds = self.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】
        k = inputs_embeds.shape[0]
        if embed_prompt:
            prompt = mlp(self.model.model.prompt)
        else:
            prompt = self.model.model.prompt
        
        
        '''
        增量添加prompt代码处
        '''
        if progressive:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          self.previous_prompts.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)
            full_prefix_len = self.previous_prompts.shape[0] + prompt.shape[0] # prefix including all previous tasks
        else:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)
            full_prefix_len = prompt.shape[0]
            
        source_mask_updated = torch.concat((torch.tensor(1).to("cuda").repeat(k,full_prefix_len),
                                             batch["attention_mask"]), axis=1)

        lm_labels = torch.concat((lm_labels[0][0].repeat(k,inputs_embeds.shape[1]-lm_labels.shape[1]),lm_labels),axis=1)

        
        '''
        encoder_outputs = model.encoder(
                                attention_mask=source_mask_updated,
                                inputs_embeds=inputs_embeds,
                                head_mask=None,  
                                output_attentions=None,  
                                output_hidden_states=None, 
                                return_dict=None,  
                            )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=source_mask_updated, 
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        
        outputs = self.model.generate(inputs.input_ids, max_length=1024)
        '''
        outputs = self.model(inputs_embeds=inputs_embeds, labels=lm_labels, attention_mask=source_mask_updated)
        loss = outputs[0]

        return loss




    # Freeze model weights
    def do_freeze_weights(self):
        model = self.model
        for name, param in model.named_parameters():
            param.requires_grad = False


    # Freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
    def freeze_unfreeze_mlps(self, tasks, requires_grad=False):
        assert self.prefix_MLP != None

        for task_num in tasks:
            #for name, param in self.prefix_MLPs[t].named_parameters():
            for name, param in self.model.model.mlps[task_num].named_parameters():
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    param.grad = None # remove old gradient

    
    # Perform training on a single task
    def train_one_task(self,
                       task,
                       task_num,
                       epochs,
                       progressive=True):
        
        #将新的prompt加入优化器
        # old_prompt = deepcopy(self.model.model.prompt)
        print('task = ', task)
        if progressive:
            assert self.prefix_len>0 # can only do progressive prompts when prompt tuning
            print('progressive prompts')
        if self.early_stopping:
            self.best_acc = 0.0 # re-setting best acc

        if self.prefix_MLP!=None:
            print('Freezing all MLPs except for ', task)
            mlp = self.model.model.mlps[task_num]
            self.freeze_unfreeze_mlps([x for x in range(len(self.train_task_list)) if x!=task_num], requires_grad=False)
            self.freeze_unfreeze_mlps([task_num], requires_grad=True) # unfreezing current task

        model = self.model
        model.model.prompt.requires_grad=True
        # with torch.no_grad():
        #     # prompts = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
        #     #                             requires_grad=True))
        #     model.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
        #                                 requires_grad=True))
        # model.model.prompt.requires_grad = True
        # model.model.to(self.device)

        dataloader_train = self.train_task_list[task]
        total_steps = self.args.num_train_epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        val_acc = []

        for epoch in range(epochs):
            print(epoch)
            model.train()
            if self.prefix_MLP!=None:
                mlp.train()

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                if self.prefix_len>0: # prompt tuning
                    loss = self.train_step_lester(batch,
                                                  task=task if self.prefix_MLP!=None else None,
                                                  task_num=task_num if self.prefix_MLP!=None else None,
                                                  progressive=progressive)
                    
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                model.backward(loss)
                model.step()
            
                

            # evaluate accuracy after each epoch
            if self.prefix_MLP!=None:
                #生成prompt并拼接到previous_prompts
                mlp.eval()
                prompt = mlp(self.model.model.prompt)
            else:
                if self.prefix_len>0:
                    prompt = self.model.model.prompt
                    print(prompt.shape)
                else:
                    prompt = None
            # if progressive:
            #     prompt = torch.concat([prompt, self.previous_prompts], axis=0)

        if progressive:
            self.progress_previous_prompts(task=task)
            # model.model.prompt.data = deepcopy(old_prompt.data)

def convert_PP_model(model, args):
    
    def init_new_prompt(prompt_len):
        N = args.embed_tokens_length
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N) # random token
                w = deepcopy(args.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w/100)
                # prompt_weigths.append(w)

        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths
    model.model.prompt = nn.Parameter(torch.tensor(init_new_prompt(args.prefix_len),requires_grad=True))
    model.model.mlps = nn.ModuleList([ResMLP(bottleneck_size=2*args.embed_tokens_dim, emb_dimension=args.embed_tokens_dim) for _ in range(args.task_length)])
    
    return model
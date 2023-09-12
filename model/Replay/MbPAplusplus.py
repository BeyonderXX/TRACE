import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import trange
import copy
import random
from model.base_model import CL_Base_Model
from tqdm import tqdm

class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):

        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, 768)

    def push(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        input_ids, attn_masks, labels = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key
            self.memory.update(
                {key: (input_ids[i], attn_masks[i], labels[i])})

    def _prepare_batch(self, sample):
        """
        Parameter:
        sample -> list of tuple of experiences
               -> i.e, [(content_1,attn_mask_1,label_1),.....,(content_k,attn_mask_k,label_k)]
        Returns:
        batch -> tuple of list of content,attn_mask,label
              -> i.e, ([content_1,...,content_k],[attn_mask_1,...,attn_mask_k],[label_1,...,label_k])
        """
        input_ids = []
        attn_masks = []
        labels = []
        # Iterate over experiences
        for content, attn_mask, label in sample:
            # convert the batch elements into torch.LongTensor
            input_ids.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (input_ids, attn_masks, labels)

    def get_neighbours(self, eval_keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        self.all_keys = list(self.memory.keys()) #[len(keys),hidden_size]
        self.all_keys = torch.stack(self.all_keys,dim=0).to("cuda")  # [len(keys),hidden_size]

        # Iterate over all the input keys
        # to find neigbours for each of them
        for eval_key in eval_keys:  #[hidden_size]
            similarity = torch.mm(eval_key.unsqueeze(0),self.all_keys.permute(*torch.arange(self.all_keys.ndim - 1, -1, -1))) #[1,len(keys)]
            # print(similarity)
            values, indices = similarity.topk(k,dim=1)
            # neighbors_keys = self.all_keys[indices].squeeze(0)
            # converts experiences into batch
            indices = indices.squeeze(0)
            neighbors  = [list(self.memory.values())[idx] for idx in indices]  # [(input_ids, attn_masks, labels) * k]
            batch = self._prepare_batch(neighbors)  # ([],[],[])
            samples.append(batch)

        return samples  # [([],[],[]),([],[],[]),([],[],[])]
    
    def sample(self, sample_size):
        
        keys = random.sample(list(self.memory),sample_size)  #随机采样key
        input_ids = [self.memory[k][0] for k in keys]
        attn_masks = [self.memory[k][1] for k in keys]
        labels = [self.memory[k][2] for k in keys]
        
        return (input_ids, attn_masks, labels)
        


class MbPAplusplus(CL_Base_Model):
    """
    Implements Memory based Parameter Adaptation model
    """

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, L=30, K_neightbors=32, replay_size=64):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.REPLAY_FREQ = 0  #  多少step后replay
        self.L = L                      #  replay多少轮
        self.memory = ReplayMemory()
        self.K_neightbors = K_neightbors
        self.replay_size = replay_size
        self.train_batch_size = self.args.per_device_train_batch_size
        self.eval_batch_size = self.args.per_device_eval_batch_size


    def get_keys(self,batch):
        outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        keys = outputs.hidden_states[-1]  #最后一层的输出，hidden_states
        keys = keys[:, 0, :].squeeze(1) #取第一个token, (bs, hidden_size)
        
        return keys

                
    def train_one_task(self, task, i_task, epochs):
        dataloader_train = self.train_task_list[task]
        self.train_length = len(dataloader_train)
        total_steps = self.args.num_train_epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        self.REPLAY_FREQ = total_steps//4

        for epoch in range(epochs):

            self.model.train()

            # Train the data for one epoch
            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']

                # Release file descriptors which function as shared
                # memory handles otherwise it will hit the limit when
                # there are too many batches at dataloader
                # Perform sparse experience replay after every REPLAY_FREQ steps
                # if (step+1) % self.REPLAY_FREQ == 0 and i_task!=0 and len(self.memory.memory) >= self.sample_size:
                if (step+1) % self.REPLAY_FREQ == 0 and len(self.memory.memory) >= self.replay_size:
                    # sample 64 examples from memory
                    S_input_ids, S_attn_masks, S_labels = self.memory.sample(sample_size=self.replay_size)
                    
                    for i in range(self.replay_size):
                    
                        input_ids = S_input_ids[i].unsqueeze(0).to("cuda")
                        attn_masks = S_attn_masks[i].unsqueeze(0).to("cuda")
                        labels = S_labels[i].unsqueeze(0).to("cuda")

                        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attn_masks)
                        loss = outputs.loss
                        # Backward pass
                        self.model.backward(loss)
                        self.model.step()

                    del input_ids
                    del attn_masks
                    del labels
                    del loss
                # Unpacking the batch items
                batch = {k:batch[k].to('cuda') for k in batch}
                
                outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                loss = outputs.loss
                # Backward pass
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                self.model.backward(loss)
                self.model.step()
                # Get the key representation of documents
                
                #节省计算，不调用get_keys()
                keys = outputs.hidden_states[-1]  #最后一层的输出，hidden_states
                keys = keys[:, 0, :].squeeze(1) #取第一个token, (bs, hidden_size)
                # Push the examples into the replay memory
                self.memory.push(keys.detach().cpu(), (batch['input_ids'].cpu(),
                                                batch['attention_mask'].cpu(), batch['labels'].cpu()))
                
    def infer(self, input_ids, attn_mask, R_input_ids, R_attn_masks, R_labels):


        # create a local copy of the classifier network

        # Current model weights
        base_weights = list(self.model.parameters())
        # Train the adaptive classifier for L epochs with the rt_batch
        for _ in range(self.L):
            for i in range(len(R_input_ids)):
                
                R_outputs = self.model(input_ids=R_input_ids[i], attention_mask=R_attn_masks[i], labels=R_labels[i])
                R_loss = R_outputs.loss
                # Initialize diff_loss to zero and place it on the appropriate device
                diff_loss = torch.Tensor([0]).to(
                    "cuda" if torch.cuda.is_available() else "cpu")
                # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                # of their differences
                curr_weights = list(self.adaptive_model.parameters())
                for base_param, curr_param in zip(base_weights, curr_weights):
                    diff_loss += (curr_param-base_param).pow(2).sum()

                # Total loss due to log likelihood and weight restraint
                total_loss = 0.001*diff_loss + R_loss
                self.model.backward(total_loss)
                self.model.step()
        
        infer_outputs = self.model(input_ids=input_ids,  attention_mask=attn_mask)
        
        #param restore
        for idx,name, param in enumerate(self.model.named_parameters()):
            param.data.copy_(base_weights[idx])
        return infer_outputs

    def evaluate(self, round, i_task, task):
        eval_task_names = list(self.eval_task_list.keys())[:i_task+1]
        for eval_task_name in eval_task_names:
            self.evaluate_one_task(eval_task_name)
    
    def evaluate_one_task(self, task):
                
        eval_dataloader = self.eval_task_list[task]
        for step, batch in enumerate(eval_dataloader):
            del batch['sources']
            batch = {k:batch[k].to('cuda') for k in batch}
            input_ids = batch['input_ids']
            attn_masks = batch['attention_mask']
            eval_keys = self.get_keys(batch)
            neightbors_samples = self.memory.get_neighbours(eval_keys, self.K_neightbors)   #[([],[],[]),([],[],[]),([],[],[])]
            
            for input_ids, attn_mask, (rt_input_ids, rt_attn_masks, rt_labels) in tqdm(zip(input_ids, attn_masks, neightbors_samples), total=len(input_ids)):
                rt_input_ids = [r_input_ids.to("cuda") for r_input_ids in rt_input_ids]
                rt_attn_masks = [r_attn_masks.to("cuda") for r_attn_masks in rt_attn_masks]
                rt_labels = [r_labels.to("cuda") for r_labels in rt_labels]
                outputs = self.infer(input_ids, attn_mask, rt_input_ids, rt_attn_masks, rt_labels)
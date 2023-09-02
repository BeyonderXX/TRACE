import torch
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn
from tqdm.auto import tqdm



def project_vec(vec, proj_basis, gpu):
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = torch.matmul(vec, proj_basis)  # basis_size
        # out = torch.matmul(proj_basis, dots)
        # TODO : Check !!!!
        out = torch.matmul(proj_basis, dots.T)
        return out
    else:
        return torch.zeros_like(vec)


def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)

def count_parameter(model):
    return sum(p.numel() for p in model.parameters())

def orthonormalize(vectors, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in tqdm(range(start_idx, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors

class OGD:
    
    def __init__(self, model, tokenizer, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
    
    
    def get_params_dict(self):
        return self.model.parameters()
    
        
    def train_step(self,
                    batch):

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        inputs_embeds = self.model.model.embed_tokens(batch["source_ids"])  #向量，【batch * embedding_size】

        outputs = self.model(inputs_embeds=inputs_embeds, labels=lm_labels)
        loss = outputs[0]
        
        return loss
    
    
    def train_one_task(self,
                       task,
                       epochs=40):

        print('task = ', task)
        if self.early_stopping:
            self.best_acc = 0.0 # re-setting best acc



        '''
        optimizer是否需要传入?
        '''
        with torch.no_grad():
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay,
                                                task=task)
        dataloader_train = self.tasks_data_dict[task]['train']


        for epoch in range(epochs):
            print(epoch)
            self.model.train()
            
            new_basis=[]
            for i, batch in enumerate(tqdm(dataloader_train)):
                
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)

                loss.backward()
                
                #投影
                cur_param = parameters_to_vector(self.get_params_dict())
                grad_vec = parameters_to_grad_vector(self.get_params_dict())
                proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.config.gpu)
                
                new_grad_vec = grad_vec - proj_grad_vec
                cur_param -= self.config.lr * new_grad_vec
                vector_to_parameters(cur_param, self.get_params_dict())
                
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                #添加新的梯度方向，构成新的非正交基
                new_basis.append(grad_vec)
                
            #正交化
            new_basis_tensor = torch.stack(new_basis).T
            self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
            self.ogd_basis = orthonormalize(self.ogd_basis, normalize=True)


    # Train model continually
    def train_continual(self,
                        epochs=40,
                        ):

        for num, task in enumerate(self.task_list):
            self.train_one_task(task, epochs)
            
import torch


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.task_list = task_list
        self.args = args
        
        
    def train_one_task(self, task, i_task, epochs):
        pass
    
    
    def train_continual(self):
        for num, task in enumerate(self.task_list):
            self.train_one_task(task, num, self.args.num_train_epochs)
            
            
    def evaluate(self, cur_task):
        pass
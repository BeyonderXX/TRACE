import torch


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.args = args
        
        
    def train_one_task(self, task, i_task, epochs):
        pass
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, self.args.num_train_epochs)
            self.evaluate(i_task)
            
            
    def evaluate(self, i_task):
        #评估，不同的dataset对应不同的metrics
        pass
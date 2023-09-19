from model.base_model import CL_Base_Model
import os
import time
from utils.utils import print_rank_0


class lora(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)

    
    def save_model(self, i_task):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)
            

import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        
        
    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()


            # Evaluate perplexity on the validation set.
            print_rank_0(
                f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
                self.args.global_rank)
            perplexity = self.perplexity_evaluation(eval_dataloader, device)
            print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            self.model.tput_timer.update_epoch_count()
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, self.args.num_train_epochs)
            for infer_task_id, _task in enumerate(self.train_task_list):
                if infer_task_id > i_task:
                    break
                self.evaluate(i_task, infer_task_id, _task)

    def dist_results_gather(self, generate_ids, pad_token=-1):
        # (batch_size, seq_len)
        result = generate_ids  # Example tensor
        local_batch_size = torch.tensor([result.size(0)], dtype=torch.int).cuda()
        local_seq_len = torch.tensor([result.size(1)], dtype=torch.int).cuda()

        # 收集所有 GPUs 上的 batch_size 和 seq_len
        global_batch_sizes = [torch.tensor([0], dtype=torch.int).cuda() for _ in range(dist.get_world_size())]
        global_seq_len = [torch.tensor([0], dtype=torch.int).cuda() for _ in range(dist.get_world_size())]
        dist.all_gather(global_batch_sizes, local_batch_size)
        dist.all_gather(global_seq_len, local_seq_len)

        # 确定 max_seq_len
        max_seq_len = max([int(seq_len.item()) for seq_len in global_seq_len])

        # left Pad 本地的 tensor 到 (_, max_seq_len)
        if result.size(1) < max_seq_len:
            pad_seq_len = (max_seq_len - result.size(1), 0)
            result = F.pad(result, pad_seq_len, "constant", pad_token).cuda()

        # 使用 all_gather 收集所有 GPUs 上的 padded tensors
        total_results = [torch.zeros((int(bs.item()), max_seq_len), dtype=result.dtype).cuda() for bs in global_batch_sizes]
        dist.all_gather(total_results, result)

        # Flatten total_results 来创建一个大的列表
        flat_results = torch.cat(total_results, dim=0)

        return flat_results, max_seq_len
            
    def evaluate(self, round, infer_task_id, task):
        #评估，不同的dataset对应不同的metrics
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
                # prompt_len = batch['input_ids'].shape[1]

                # update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Step {step}"
                    progress_bar.set_description(description, refresh=False)

                with torch.no_grad():
                    # sft config
                    generate_ids = model.generate(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
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

    
    def save_model(self):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args)

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage)
        print_rank_0(f'Sucessful saving the final model to {self.args.output_dir}', self.args.global_rank)
        

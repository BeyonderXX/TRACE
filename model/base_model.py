import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
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
            for j_task, _task in enumerate(self.train_task_list):
                if j_task > i_task:
                    break
                self.evaluate(i_task, j_task, _task)
            
            
    def evaluate(self, round, i_task, task):
        #评估，不同的dataset对应不同的metrics
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        dataset_path = os.path.join(self.args.data_path,task)
        _, _, infer_dataset = create_prompt_dataset(
            self.args.local_rank,
            dataset_path,
            self.args.data_output_path,
            self.args.seed
        )

        inf_data_collator = DataCollator(
            self.tokenizer,
            model=self.model,
            padding="longest",
            max_prompt_len=self.args.max_prompt_len,
            max_ans_len=self.args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )

        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(infer_dataset,
                                      collate_fn=inf_data_collator,
                                      sampler=infer_sampler,
                                      batch_size=self.args.per_device_eval_batch_size)

        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))


        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            model.eval()

            for step, batch in enumerate(infer_dataloader):

                sources_sequences += batch['sources']
                del batch['sources']
                batch = to_device(batch, device)
                progress_bar.update(1)
                prompt_len = batch['input_ids'].shape[1]

                # update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Step {step}"
                    progress_bar.set_description(description, refresh=False)

                with torch.no_grad():
                    # TODO, add more inference params
                    # backbone config
                    # generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len,
                    #                               pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )

                    # sft config
                    generate_ids = model.generate(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
                                                  max_new_tokens=self.args.max_ans_len,
                                                  bos_token_id=self.tokenizer.bos_token_id,
                                                  eos_token_id=self.tokenizer.eos_token_id,
                                                  pad_token_id=self.tokenizer.unk_token_id,
                                                  generation_config=generation_config,
                                                  )

                sequences = self.tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
                predicted_sequences += sequences

            return sources_sequences, predicted_sequences


        def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                   ground_truths: list, round: int, i_task: int, task: str):
            # save as a json file
            df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                  'labels': ground_truths}
            with open(self.args.output_dir + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w", encoding='utf-8') as file:
                json.dump(df, file, ensure_ascii=False)


        # Inference !
        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences = prediction(self.model, infer_dataloader)

        with open(self.args.data_path + "/" + task + "/test.json", "r", encoding="utf-8") as file:
            testset = json.load(file)
        ground_truths = []
        for item in testset:
            ground_truths.append(item["answer"])

        # Get Accuracy/ROUGE/BLEU/...
        # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
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
        elif task == "ToolBench":
            evaluation_result = eval_ToolBench.eval(predicted_sequences, ground_truths)
        else:
            evaluation_result = {}

        if self.args.global_rank <= 0:
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, i_task, task)

    
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
        
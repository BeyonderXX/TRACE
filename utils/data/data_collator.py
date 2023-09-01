import logging
import torch
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True       # ‘longest’
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs

    # support decoder-only models for left padding
    def decoder_call(self, batch, return_tensors):
        sources = []
        prompt_lens = []  # 用于存储每个prompt的长度
        label_lens = []  # 用于存储每个label的长度
        actual_max_len = 0  # 用于存储batch中的实际最大长度

        for instance in batch:
            instruction = instance['prompt']
            label = instance['answer']

            # add bos and eos
            task_input = self.tokenizer.bos_token + instruction
            label = label + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(task_input, add_special_tokens=False,return_tensors=return_tensors)["input_ids"][0]
            tokenized_label = self.tokenizer(label, add_special_tokens=False, return_tensors=return_tensors)["input_ids"][0]

            # 保存prompt的长度
            prompt_lens.append(len(tokenized_input))
            label_lens.append(len(tokenized_label))

            # 根据是否是inference来计算total_len
            if not self.inference:
                total_len = len(tokenized_input) + len(tokenized_label)
                sources.append(task_input + label)
            else:
                total_len = len(tokenized_input)
                sources.append(task_input)

            if total_len > actual_max_len:
                actual_max_len = total_len

        # 取batch中的最大长度和limit_input_len中的最小值作为实际padding长度
        # 并确保长度是pad_to_multiple_of的倍数
        limit_input_len = self.max_prompt_len if self.inference else self.max_prompt_len + self.max_ans_len
        actual_pad_len = min(actual_max_len, limit_input_len)
        remainder = actual_pad_len % self.pad_to_multiple_of
        if remainder > 0:
            actual_pad_len += self.pad_to_multiple_of - remainder

        model_inputs = self.tokenizer(
            sources,
            max_length=actual_pad_len,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            truncation=True
        )

        # 对于left padding和prompt部分的mask
        if not self.inference:
            labels_tensor = model_inputs["input_ids"].clone()
            for idx, (prompt_len, label_length) in enumerate(zip(prompt_lens, label_lens)):
                padding_len = actual_pad_len - (prompt_len + label_length)
                labels_tensor[idx, :padding_len + prompt_len] = self.label_pad_token_id
            model_inputs['labels'] = labels_tensor
        model_inputs['sources'] = sources

        return model_inputs


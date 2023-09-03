import json
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy


# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        # answer
        if datium.find("Answer: ") == -1: # not found /format error
            answer = ""
        else:
            answer = datium[datium.find("Answer: ") + 8: datium.find("Answer: ") + 9] # one char
        answers.append(answer)
        # reasoning
        if datium.find("Because ") == -1: # not found /format error
            reasoning = ""
        else:
            reasoning = datium[datium.find("Because ") + 8: len(datium)]
        reasonings.append(reasoning)
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result

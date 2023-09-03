## 默认模型在测试时输出的顺序和测试集顺序相同
## The inputs must not be shuffled during evaluation.


import re
from rouge import Rouge
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)
    avg_rouge = sum(rouges) / len(results)
    return avg_rouge


########################
## Accuracy (EM)
########################
def caculate_accuracy(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        if prediction == target:
            scores += 1
    avg_score = scores / len(results)
    return avg_score


########################
## F1-micro
########################
def f1_score(list1, list2):
    # TP: item in list1 and list2
    # FP: item in list1 but not in list2
    # TN: item not in list1 and list2
    # FN: item in list2 but not in list1
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1)
    recall = num_TP / len(list2)
    return 2 * (precision * recall / (precision + recall))


def caculate_f1(results, data):
    scores = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if len(prediction) == 0 or len(target) == 0:
            continue
        score = f1_score(target, prediction)
        scores.append(score)
    avg_score = sum(scores) / len(results)
    return avg_score


########################
## fuzzywuzzy
########################
def caculate_fuzz(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        scores += fuzz.ratio(prediction, target)
    avg_score = scores / len(results) 
    return avg_score

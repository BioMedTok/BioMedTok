# SubwordAnalysis.py

import json
from collections import Counter
from re import A

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, CamembertTokenizer, RobertaTokenizerFast

f_in = open("./francais.txt","r")
# f_in = open("./DataDecoupageSpecialites.txt","r")
morphemes = [e for e in f_in.read().split("\n") if len(e) >= 3]
f_in.close()

models = [
    "BioMedTok/BPE-HF-NACHOS-FR",
    "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",

    "BioMedTok/BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-PubMed-FR-Morphemes",
    
    "BioMedTok/BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-CC100-FR-Morphemes",
    
    "BioMedTok/BPE-HF-Wikipedia-FR",
    "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-CC100-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-Wikipedia-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes",
]

def PROCESS(text):
    text = text.replace("▁","").replace("Ġ","").replace("Ã©","é")
    text = text.replace("Ã¯","ï").replace("Ã¨","è").replace("Ã","à")
    text = text.replace("Ã´","ô").replace("Ã§","ç").replace("Ãª","ê")
    text = text.replace("Ã¹","ù").replace("Å","œ").replace("Ã«","ë")
    text = text.replace("Ã¦","æ").replace("Ã¼","ü").replace("Ã¢","â")
    text = text.replace("â¬","€").replace("Â©","©").replace("Â¤","¤")
    text = text.replace("à®","î").replace("à¢","â").replace("à´","ô")
    text = text.replace("à»","û").replace("à§","ç").replace("à«","ë")
    text = text.replace("à¼","ü")
    return text

results = {model_name: {_morpheme.split(";")[0]: "" for _morpheme in morphemes} for model_name in models}

results_metrics = {model_name: {} for model_name in models}

for model_name in models:

    if "sentencepiece" in model_name.lower():
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
    elif "bpe_tokenizers" in model_name.lower() and "-roberta" in model_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    elif "bpe_tokenizers" in model_name.lower():
        m_name = model_name.split("/")[-1]
        f_vocab = f"{model_name}/{m_name}-vocab.json"
        f_merges = f"{model_name}/{m_name}-merges.txt"
        tokenizer = ByteLevelBPETokenizer(f_vocab, f_merges)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    _local_acc = []
    _local_same_nbr_tokens = []
    _local_is_lower = []

    for _morpheme in morphemes:

        _base, _spe, _target = _morpheme.split(";")
        _target = _target.split("|")
        
        text = _base

        if "bpe_tokenizers" in model_name.lower() and not "-roberta" in model_name.lower():
            subwords = tokenizer.encode(text)
            subwords = tokenizer.decode(subwords.ids, skip_special_tokens=False)
        else:
            subwords = tokenizer.tokenize(text)

        subwords = [PROCESS(_t) for _t in subwords if len(PROCESS(_t)) > 0]

        if len(subwords) == 1:
            is_same = True
            length_good = True
        else:
            is_same = subwords == _target
            length_good = len(subwords) == len(_target)

            if length_good == False:
                is_lower = len(subwords) < len(_target)
                _local_is_lower.append(is_lower)

        results[model_name][_base] = {
            "target": _target,
            "segmentation": subwords,
            "is_same": is_same,
        }

        _local_acc.append(is_same)
        _local_same_nbr_tokens.append(length_good)
    
    avg_acc = (sum(_local_acc) / len(_local_acc)) * 100
    avg_acc_length = (sum(_local_same_nbr_tokens) / len(_local_same_nbr_tokens)) * 100

    percent_lower = (sum(_local_is_lower) / len(_local_same_nbr_tokens)) * 100
    percent_higher = ((len(_local_is_lower) - sum(_local_is_lower)) / len(_local_same_nbr_tokens))* 100

    print(model_name, " & ", "{:.2f}".format(avg_acc), " & ", "{:.2f}".format(avg_acc_length), " & ", "{:.2f}".format(percent_lower), " & ", "{:.2f}".format(percent_higher))

    results_metrics[model_name]["avg_acc"] = avg_acc
    results_metrics[model_name]["avg_acc_length"] = avg_acc_length
    results_metrics[model_name]["percent_lower"] = percent_lower
    results_metrics[model_name]["percent_higher"] = percent_higher

with open("results_subwords_fr.json", 'w') as f:
# with open("results_subwords.json", 'w') as f:
    json.dump(results, f, indent=4)

with open("results_metrics_fr.json", 'w') as f:
# with open("results_metrics.json", 'w') as f:
    json.dump(results_metrics, f, indent=4)

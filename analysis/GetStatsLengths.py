import json

import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModelForMaskedLM

models = [

    "BioMedTok/BPE-HF-NACHOS-FR",
    "BioMedTok/BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-Wikipedia-FR",

    "BioMedTok/SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR",

    "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",
    "BioMedTok/BPE-HF-PubMed-FR-Morphemes",
    "BioMedTok/BPE-HF-CC100-FR-Morphemes",
    "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes",

    "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes",

]

all_res = {}

def MOD(text):
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

for MODEL_NAME in models:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    vocab = [MOD(k) for k in tokenizer.vocab.keys() if len(MOD(k)) > 0]
    
    all_res[MODEL_NAME] = {}

    for v in vocab:

        if len(v) not in all_res[MODEL_NAME]:
            all_res[MODEL_NAME][len(v)] = 0
        
        all_res[MODEL_NAME][len(v)] += 1

with open("all_lenghts.json", "w") as outfile:
    json.dump(all_res, outfile, indent=4)

# print(max([len(_) for _ in morphemes]))

all_lenght = []

for a in all_res:
    all_lenght.extend(all_res[a].keys())

all_lenght = sorted(list(set(all_lenght)))
print(all_lenght)
print(all_lenght[-1])

for a in all_res:
    
    _l = sorted(list(all_res[a].keys()))
    print(f"{a} => {_l}")
    
    x = []
    for l in range(1,26):

        if l in all_res[a]:
            x.append(all_res[a][l])
        else:
            x.append(0)
    
    x = np.array(x)
    print(x)
    y = np.array(range(1,26))
    print(y)

    plt.ylim(0, 6000)
    plt.bar(y, x, color="#A9C5F8")
    plt.savefig(f"./histograms/{a.replace('/','_')}")

    plt.clf()
    print("*"*50)

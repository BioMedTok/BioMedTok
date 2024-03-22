import json
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

models = [

    "BioMedTok/BPE-HF-Wikipedia-FR",
    "BioMedTok/BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-NACHOS-FR",

    "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes",
    "BioMedTok/BPE-HF-PubMed-FR-Morphemes",
    "BioMedTok/BPE-HF-CC100-FR-Morphemes",
    "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",

    "BioMedTok/SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR",
    "BioMedTok/SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR",

    "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes",
]

coverage = {m: {m2: -1 for m2 in models} for m in models}

for model_name_1 in models:

    print(model_name_1)
    
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    # vocabulary_1 = tokenizer_1.vocab
    vocabulary_1 = list(set([v.replace("Ġ","") for v in tokenizer_1.vocab]))

    for model_name_2 in models:

        print(model_name_2)
        
        tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
        # vocabulary_2 = tokenizer_2.vocab
        vocabulary_2 = list(set([v.replace("Ġ","") for v in tokenizer_2.vocab]))

        taux = len(set(vocabulary_1)&set(vocabulary_2)) / float(len(set(vocabulary_1) | set(vocabulary_2))) * 100

        # In common between both
        # set_vocabulary_2 = set(vocabulary_2)
        # in_common = len([x for x in vocabulary_1 if x in set_vocabulary_2])
        # total_uniq = len(list(set(vocabulary_1 + vocabulary_2)))
        # taux = (in_common / total_uniq) * 100

        # taux = len(set(vocabulary_1)&set(vocabulary_2)) / float(len(set(vocabulary_1) | set(vocabulary_2))) * 100

        print(f"{model_name_1} - {model_name_2} : {taux}")
        coverage[model_name_1][model_name_2] = int(taux)
        # coverage[model_name_1][model_name_2] = "%.1f" % taux

with open("coverage.json", "w") as f_coverage:
    json.dump(coverage, f_coverage, indent=4)

matrix = [[coverage[m1][m2] for m2 in models] for m1 in models]

mask = np.zeros_like(matrix)
for i in range(len(mask)):
    for j in range(i+1, len(mask)):
        mask[i,j] = 1 
mask = np.array(mask, dtype=np.bool)
print(mask)

rotation_angle = 90

cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)

cmap = "Blues"

sb.set(font_scale=0.5, rc={'axes.facecolor':'#ffffff', 'figure.facecolor':'#ffffff'})

models = [m.replace("BioMedTok/","").replace("-FR","").replace("-HF","").replace("Wikipedia","Wiki").replace("SentencePieceBPE","SP").replace("-Morphemes","+M") for m in sorted(models)]
heat_map = sb.heatmap(matrix, mask=mask, cmap=cmap, annot=True, cbar=False, fmt='g', cbar_kws={'label': 'Percentage of tokens in commons', 'orientation': 'horizontal'})
heat_map.set_yticklabels(models, rotation=0, fontsize=8)
heat_map.set_xticklabels(models, rotation=rotation_angle, fontsize=8)

plt.savefig(f"./matrix_{cmap}.png", bbox_inches='tight', dpi=600)
        
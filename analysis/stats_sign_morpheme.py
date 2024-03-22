import os
import json
from glob import glob
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from itertools import combinations

with open('./results.json', 'r') as f:
  res_all = json.load(f)

# with open('./results_DEFT2021_essai_cas_pos.json', 'r') as f:
#   res_essai_cas = json.load(f)

# with open('./results_DEFT2021_CLS.json', 'r') as f:
#   res_deft2021 = json.load(f)


models = ["../../../models/biomedtok_sentencepiecebpe-cc100-fr",
"../../../models/biomedtok_sentencepiecebpe-wikipedia-fr",
"../../../models/biomedtok_bpe-hf-nachos-fr",
"../../../models/biomedtok_bpe-hf-nachos-fr-morphemes",
"../../../models/biomedtok_sentencepiecebpe-nachos-fr",
"../../../models/biomedtok_bpe-hf-wikipedia-fr-morphemes",
"../../../models/biomedtok_sentencepiecebpe-pubmed-fr",
"../../../models/biomedtok_bpe-hf-cc100-fr-morphemes",
"../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes",
"../../../models/biomedtok_bpe-hf-wikipedia-fr",
"../../../models/biomedtok_bpe-hf-pubmed-fr",
"../../../models/biomedtok_sentencepiecebpe-cc100-fr-morphemes",
"../../../models/biomedtok_bpe-hf-pubmed-fr-morphemes",
"../../../models/biomedtok_sentencepiecebpe-pubmed-fr-morphemes",
"../../../models/biomedtok_sentencepiecebpe-wikipedia-fr-morphemes",
"../../../models/biomedtok_bpe-hf-cc100-fr"]

fewshot = "1.0"

# all
_t = ["cas|pos", "cas|ner-neg", "cas|cls", 'cas|ner-spec', "essai|pos", "essai|cls", "essai|ner-neg",
"essai|ner-spec", "quaero|ner-emea", "quaero|ner-medline", "e3c|ner-clinical",
"e3c|ner-temporal", "morfitt|cls", "frenchmedmcqa|mcqa", "frenchmedmcqa|cls", "mantragsc|ner-fr_emea",
"mantragsc|ner-fr_medline", "mantragsc|ner-fr_patents", "clister|regr", "deft2020|regr",
"deft2020|cls", "deft2021|cls", "deft2021|ner", "diamed|cls", "pxcorpus|ner", "pxcorpus|cls", "deft2019|ner"]

# sans DEFT 2021
# _t = ["cas|pos", "essai|pos", "quaero|ner-emea", "quaero|ner-medline", "e3c|ner-clinical",
# "e3c|ner-temporal", "morfitt|cls", "frenchmedmcqa|mcqa", "frenchmedmcqa|cls", "mantragsc|ner-fr_emea",
# "mantragsc|ner-fr_medline", "mantragsc|ner-fr_patents", "clister|regr", "deft2020|regr",
# "deft2020|cls", "diamed|cls", "pxcorpus|ner", "pxcorpus|cls"]

tasks = [t+"|"+fewshot for t in _t]

dict_metrics = {"cas|pos"+"|"+fewshot: 'overall_f1',
	"cas|ner-neg"+"|"+fewshot: 'overall_f1',
	"cas|ner-spec"+"|"+fewshot: 'overall_f1',
	"cas|cls"+"|"+fewshot: 'weighted_f1',
	"essai|pos"+"|"+fewshot: 'overall_f1',
	"essai|ner-neg"+"|"+fewshot: 'overall_f1',
	"essai|ner-spec"+"|"+fewshot: 'overall_f1',
	"essai|cls"+"|"+fewshot: 'weighted_f1',
	"quaero|ner-emea"+"|"+fewshot: 'overall_f1',
	"quaero|ner-medline"+"|"+fewshot: 'overall_f1',
	"e3c|ner-clinical"+"|"+fewshot: 'overall_f1',
	"e3c|ner-temporal"+"|"+fewshot: 'overall_f1',
	"morfitt|cls"+"|"+fewshot: "weighted_f1",
	"frenchmedmcqa|mcqa"+"|"+fewshot: "hamming_score",
	"frenchmedmcqa|cls"+"|"+fewshot: 'weighted_f1',
	"mantragsc|ner-fr_emea"+"|"+fewshot: "overall_f1",
	"mantragsc|ner-fr_medline"+"|"+fewshot: "overall_f1",
	"mantragsc|ner-fr_patents"+"|"+fewshot: "overall_f1",
	"clister|regr"+"|"+fewshot: 'edrm',
	"deft2019|ner"+"|"+fewshot: "overall_f1",
	"deft2020|regr"+"|"+fewshot: "edrm",
	"deft2020|cls"+"|"+fewshot: "weighted_f1",
	"deft2021|cls"+"|"+fewshot: "weighted_f1",
	"deft2021|ner"+"|"+fewshot: "overall_f1",
	"diamed|cls"+"|"+fewshot: "weighted_f1",
	"pxcorpus|ner"+"|"+fewshot: "overall_f1",
	"pxcorpus|cls"+"|"+fewshot: "weighted_f1"}

# Pour rename les modèles dans l'output
dict_rename = {
"../../../models/biomedtok_bpe-hf-nachos-fr": "BPE-NACHOS",
"../../../models/biomedtok_bpe-hf-pubmed-fr": "BPE-PubMed",
"../../../models/biomedtok_bpe-hf-cc100-fr": "BPE-CC100",
"../../../models/biomedtok_bpe-hf-wikipedia-fr": "BPE-Wiki",
"../../../models/biomedtok_sentencepiecebpe-pubmed-fr": "SP-PubMed",
"../../../models/biomedtok_sentencepiecebpe-wikipedia-fr": "SP-Wiki",
"../../../models/biomedtok_sentencepiecebpe-cc100-fr": "SP-CC100",
"../../../models/biomedtok_sentencepiecebpe-nachos-fr": "SP-NACHOS",
"../../../models/biomedtok_bpe-hf-nachos-fr-morphemes": "BPE-NACHOS+M",
"../../../models/biomedtok_bpe-hf-pubmed-fr-morphemes": "BPE-PubMed+M",
"../../../models/biomedtok_bpe-hf-cc100-fr-morphemes": "BPE-CC100+M",
"../../../models/biomedtok_bpe-hf-wikipedia-fr-morphemes": "BPE-Wiki+M",
"../../../models/biomedtok_sentencepiecebpe-pubmed-fr-morphemes": "SP-PubMed+M",
"../../../models/biomedtok_sentencepiecebpe-wikipedia-fr-morphemes": "SP-Wiki+M",
"../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes": "SP-NACHOS+M",
"../../../models/biomedtok_sentencepiecebpe-cc100-fr-morphemes": "SP-CC100+M",
}

# Liste du best modèle pour chaque tâche (modèle de référence pour le T test)
best_models = {"cas|pos"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-wikipedia-fr-morphemes',
	"cas|ner-neg"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-nachos-fr',
	"cas|ner-spec"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr',
	"cas|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-cc100-fr',
	"essai|pos"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-wikipedia-fr',
	"essai|ner-neg"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-wikipedia-fr-morphemes',
	"essai|ner-spec"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-wikipedia-fr',
	"essai|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-pubmed-fr',
	"quaero|ner-emea"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-nachos-fr',
	"quaero|ner-medline"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr',
	"e3c|ner-clinical"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes',
	"e3c|ner-temporal"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-wikipedia-fr-morphemes',
	"morfitt|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr',
	"frenchmedmcqa|mcqa"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-pubmed-fr',
	"frenchmedmcqa|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-cc100-fr-morphemes',
	"mantragsc|ner-fr_emea"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-cc100-fr',
	"mantragsc|ner-fr_medline"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-cc100-fr',
	"mantragsc|ner-fr_patents"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-wikipedia-fr-morphemes',
	"clister|regr"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-cc100-fr',
	"deft2019|ner"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-nachos-fr-morphemes',
	"deft2020|regr"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes',
	"deft2020|cls"+"|"+fewshot: '../../../models/biomedtok_bpe-hf-nachos-fr',
	"deft2021|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes',
	"deft2021|ner"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr',
	"diamed|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr',
	"pxcorpus|ner"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-nachos-fr-morphemes',
	"pxcorpus|cls"+"|"+fewshot: '../../../models/biomedtok_sentencepiecebpe-wikipedia-fr-morphemes',
}

all_resultats = {}

for task in tasks:

	all_resultats[task] = {}

	for model in models:
		all_resultats[task][model] = res_all[model][task][dict_metrics[task]] 

for task in tasks:
	# all_pairs = list(combinations(list(all_resultats[task].keys()), 2))
	# all_pairs = [('../../../models/camembert-base', i) for i in models if i != '../../../models/camembert-base']
	all_pairs = [(best_models[task], i) for i in models if i != best_models[task]]
	print('###'+task+'###')
	for paire in all_pairs:
		t_statistic, p_value = ttest_ind(all_resultats[task][paire[0]], all_resultats[task][paire[1]])
		if p_value < 0.01:
			print_tf = 'Very True'
		elif p_value < 0.05:
			print_tf = 'True'
		else:
			print_tf = 'False'
		print(dict_rename[paire[0]], "\t---\t", dict_rename[paire[1]], '\t---\t p-value : ', round(p_value, 4), '\t---\t', print_tf)
	print('\n\n')


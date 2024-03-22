import os
import json

import numpy as np

import evaluate
from datasets import load_dataset

# Identifier les runs grace au directory du path pour les corpus et au -<subset>- pour les tâches
# Mapper à la main les corpus / subset directement aux données d'OOV : datasets["PxCorpus"]["CLS"] = dataset["test"].ner_tags
# Calculer les metriques pour chaque : 4 runs / N models

metric  = evaluate.load("/users/ylabrak/TokenizationDrBERT/runs-V3-morphemes-included/metrics/seqeval.py")
print("> Metric SeqEval loaded!")

print("> Start loading datasets")

datasets_oov = {
    ('essai', 'ner_neg-ner_neg'): load_dataset("BioMedTok/ESSAI","ner_neg",data_dir="/users/ylabrak/drbenchmark_full/recipes/essai/data")["test"]["is_oov"],
    ('essai', 'pos-pos'): load_dataset("BioMedTok/ESSAI","pos",data_dir="/users/ylabrak/drbenchmark_full/recipes/essai/data")["test"]["is_oov"],
    ('essai', 'ner_spec-ner_spec'): load_dataset("BioMedTok/ESSAI","ner_spec",data_dir="/users/ylabrak/drbenchmark_full/recipes/essai/data")["test"]["is_oov"],
    
    ('cas', 'ner_neg-ner_neg'): load_dataset("BioMedTok/CAS", 'ner_neg',data_dir="/users/ylabrak/drbenchmark_full/recipes/cas/data")["test"]["is_oov"],
    ('cas', 'ner_spec-ner_spec'): load_dataset("BioMedTok/CAS", 'ner_spec',data_dir="/users/ylabrak/drbenchmark_full/recipes/cas/data")["test"]["is_oov"],
    ('cas', 'pos-pos'): load_dataset("BioMedTok/CAS", 'pos',data_dir="/users/ylabrak/drbenchmark_full/recipes/cas/data")["test"]["is_oov"],

    ('mantragsc', 'fr_medline-fr_medline'): load_dataset("BioMedTok/MANTRAGSC", 'fr_medline')["test"]["is_oov"],
    ('mantragsc', 'fr_patents-fr_patents'): load_dataset("BioMedTok/MANTRAGSC", 'fr_patents')["test"]["is_oov"],
    ('mantragsc', 'fr_emea-fr_emea'): load_dataset("BioMedTok/MANTRAGSC", 'fr_emea')["test"]["is_oov"],

    ('pxcorpus', 'ner-None'): load_dataset("BioMedTok/PxCorpus")["test"]["is_oov"],

    ('e3c', 'ner-French_clinical'): load_dataset("BioMedTok/E3C","French_clinical")["test"]["is_oov"],
    ('e3c', 'ner-French_temporal'): load_dataset("BioMedTok/E3C","French_temporal")["test"]["is_oov"],

    ('quaero', 'ner-emea'): load_dataset("BioMedTok/QUAERO","emea")["test"]["is_oov"],
    ('quaero', 'ner-medline'): load_dataset("BioMedTok/QUAERO","medline")["test"]["is_oov"],

    ('deft2019', 'ner-None'): load_dataset("BioMedTok/DEFT2019",data_dir="/users/ylabrak/drbenchmark_full/recipes/deft2019/data")["test"]["is_oov"],

    ('deft2021', 'ner-ner'): load_dataset("BioMedTok/DEFT2021","ner",data_dir="/users/ylabrak/drbenchmark_full/recipes/deft2021/data")["test"]["is_oov"],
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

print("> Datasets loaded!")

all_results = {}

all_combinaisons = []

for corpus_name in os.listdir("/users/ylabrak/TokenizationDrBERT/runs-V3-morphemes-included/archived_runs_merged_v3_(no morphemes)+v5_(with morphemes)/runs/"):

    print(f"> {corpus_name}")

    path = f"/users/ylabrak/TokenizationDrBERT/runs-V3-morphemes-included/archived_runs_merged_v3_(no morphemes)+v5_(with morphemes)/runs/{corpus_name}/runs/"

    for file_name in os.listdir(path):

        splitted = file_name.split("-")

        if len(splitted) < 2:
            continue

        print(f"> \t {file_name}")

        f = open(path + file_name,"r")
        data = json.load(f)
        f.close()

        subset = splitted[2] + "-" + str(data["hyperparameters"]["subset"])

        if ("cls" in subset.lower()) or ("regression" in subset.lower()) or ("mcqa" in subset.lower()):
            continue

        model_name = data["hyperparameters"]["model_name"].split("/")[-1]

        _samples = {
            "contains_oov": {
                "real": [],
                "pred": [],
            },
            "doesnt_contains_oov": {
                "real": [],
                "pred": [],
            },
        }

        for c_real, c_pred, c_oov in zip(data["predictions"]["real_labels"], data["predictions"]["system_predictions"], datasets_oov[(corpus_name, subset)]):

            # If contains an OOV
            if sum(c_oov) > 0:
                _samples["contains_oov"]["real"].append(c_real)
                _samples["contains_oov"]["pred"].append(c_pred)
            else:
                _samples["doesnt_contains_oov"]["real"].append(c_real)
                _samples["doesnt_contains_oov"]["pred"].append(c_pred)

        _results = {
            "contains_oov": metric.compute(predictions=_samples["contains_oov"]["pred"], references=_samples["contains_oov"]["real"]),
            "doesnt_contains_oov": metric.compute(predictions=_samples["doesnt_contains_oov"]["pred"], references=_samples["doesnt_contains_oov"]["real"]),
        }
        print(_results)

        key = f"{model_name}|{corpus_name}|{subset}"

        if key not in all_results:
            all_results[key] = []
        
        all_results[key].append(_results)
        
        all_combinaisons.append((corpus_name, subset))

all_combinaisons = list(set(all_combinaisons))
print(all_combinaisons)

with open("./results_sentence_oov_v5.json", 'w') as f:
    json.dump(all_results, f, indent=4, cls=NpEncoder)


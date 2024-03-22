import json

import numpy as np
 
f = open("average_results_sentence_oov.json","r")
data = json.load(f)
f.close()

results = {}

for d in data:

    model_name, corpus_name, subset_name = d.split("|")

    short_m_name = model_name.replace("-morphemes","")

    key = f"{corpus_name}|{subset_name}"

    if key not in results:
        results[key] = {}

    if short_m_name not in results[key]:
        results[key][short_m_name] = {
            "morphemes": -1,
            "no_morphemes": -1,
        }
    
    # If contains morpheme in the name of the model
    if short_m_name != model_name:
        results[key][short_m_name]["morphemes"] = data[d]
    else:
        results[key][short_m_name]["no_morphemes"] = data[d]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open("./arranged_average_results_sentence_oov.json", 'w') as f:
    json.dump(results, f, indent=4, cls=NpEncoder)

##################################
####      GET BETTER          ####
##################################

print("#"*50)
print("ARE BETTER:")
print("#"*50)

cpt_better = 0
cpt_total = 0

all_corpus = []
all_models = []

for corpus_subset in results:

    for model_spec in results[corpus_subset]:

        e = results[corpus_subset][model_spec]

        cpt_total += 1

        if e["morphemes"]["contains_oov"] > e["no_morphemes"]["contains_oov"]:
            cpt_better += 1
            print(f"> {corpus_subset} - {model_spec} : M {e['morphemes']['contains_oov']*100} > NM {e['no_morphemes']['contains_oov']*100} from {e['morphemes']['contains_oov']*100 - e['no_morphemes']['contains_oov']*100}")
            all_corpus.append(corpus_subset)
            all_models.append(model_spec)

print(f">> CPT : {cpt_better} / {cpt_total} => {(cpt_better / cpt_total) * 100} <<")

print(">> all_corpus")
print(list(set(all_corpus)))
print(len(list(set(all_corpus))))

print(">> all_models")
print(list(set(all_models)))
print(len(list(set(all_models))))

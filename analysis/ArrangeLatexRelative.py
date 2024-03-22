import json

import numpy as np
 
f = open("average_results_sentence_oov.json","r")
data = json.load(f)
f.close()

results = {}

all_models_names = []

for d in data:

    model_name, corpus_name, subset_name = d.split("|")

    short_m_name = model_name.replace("-morphemes","")

    all_models_names.append(short_m_name)

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

all_models_names = [
    'biomedtok_bpe-hf-nachos-fr',
    'biomedtok_bpe-hf-pubmed-fr',
    'biomedtok_bpe-hf-cc100-fr',
    'biomedtok_bpe-hf-wikipedia-fr',
    'biomedtok_sentencepiecebpe-nachos-fr',
    'biomedtok_sentencepiecebpe-pubmed-fr',
    'biomedtok_sentencepiecebpe-cc100-fr',
    'biomedtok_sentencepiecebpe-wikipedia-fr',
]
# all_models_names = list(set(all_models_names))

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

all_gains = []
all_loss = []

_res_avg_model = {}

_table = []

_table.append(" Tasks & " + " & ".join([md.replace("biomedtok_","") for md in all_models_names]))

_res_avg_model = {}
for amn in all_models_names:
    _res_avg_model[amn] = []

_res_avg_tasks = {}
for corpus_subset in results:
    _res_avg_tasks[corpus_subset] = []

for corpus_subset in results:

    if "deft2019" in corpus_subset:
        continue

    _local_res = []

    for model_spec in all_models_names:
    # for model_spec in results[corpus_subset]:

        e = results[corpus_subset][model_spec]

        cpt_total += 1

        diff_1 = (e['morphemes']['contains_oov']*100) - (e['no_morphemes']['contains_oov']*100)
        diff_2 = diff_1 / (e['no_morphemes']['contains_oov']*100)
        diff = diff_2 * 100
        diff_str = str(round(abs(diff), 2))
        value_1 = str(round(e['morphemes']['contains_oov']*100, 2))

        _res_avg_model[model_spec].append(diff)
        _res_avg_tasks[corpus_subset].append(diff)

        if diff > 0:
            cpt_better += 1
            print(f"> {corpus_subset} - {model_spec} : M {e['morphemes']['contains_oov']*100} > NM {e['no_morphemes']['contains_oov']*100} from {diff}")
            all_corpus.append(corpus_subset)
            all_models.append(model_spec)
            _local_res.append(value_1 + " $\\uparrow$ \\scalebox{0.66}{" + diff_str + "} ")
            all_gains.append(diff)
        else:
            print(f"# LOWER # {corpus_subset} - {model_spec} : diff => {diff}")
            _local_res.append(value_1 + " $\\downarrow$ \\scalebox{0.66}{" + diff_str + "} ")
            all_loss.append(abs(diff))
        
    _table.append(corpus_subset.replace("_","\_").replace("|","-") + " & " + " & ".join(_local_res))

print(f">> CPT : {cpt_better} / {cpt_total} => {(cpt_better / cpt_total) * 100} <<")

print(">> all_corpus")
print(list(set(all_corpus)))
print(len(list(set(all_corpus))))

print(">> all_models")
print(list(set(all_models)))
print(len(list(set(all_models))))

print("#"*50)

print(" \\\\ \\hline \n".join(_table))

print("#"*50)

print(f">> Average gain: {sum(all_gains) / len(all_gains)}")
print(f">> Min gain: {min(all_gains)}")
print(f">> Max gain: {max(all_gains)}")

print(f">> Average loss: {sum(all_loss) / len(all_loss)}")
print(f">> Min loss: {min(all_loss)}")
print(f">> Max loss: {max(all_loss)}")

all_values = all_gains + [-v for v in all_loss]
print(f"Overall: {sum(all_values) / len(all_values)}")

print("#"*25)
print("MODELS")
print("#"*25)

for amn in all_models_names:
    print(amn)
    print(sum(_res_avg_model[amn]) / len(_res_avg_model[amn]))
    print()

print("#"*25)
print("SUBSETS")
print("#"*25)

for corpus_subset in results:

    if "deft2019" in corpus_subset:
        continue
    
    print(corpus_subset)
    print(sum(_res_avg_tasks[corpus_subset]) / len(_res_avg_tasks[corpus_subset]))
    print()

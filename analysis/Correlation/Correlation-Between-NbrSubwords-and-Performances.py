import json
from scipy.stats.stats import pearsonr   

def getPerformances():

    res_map = {}

    f_scores = open("./overall_averaged_metrics.json")
    data_scores = json.load(f_scores)
    f_scores.close()

    for model_name in data_scores:

        model_name = model_name.lower()

        if "sentencepiece" in model_name:
            _algo = "sentencepiece"
        elif "bpe-hf" in model_name:
            _algo = "bpe"

        if "cc100" in model_name:
            _data = "cc100"
        elif "nachos" in model_name:
            _data = "nachos"
        elif "pubmed" in model_name:
            _data = "pubmed"
        elif "wiki" in model_name:
            _data = "wiki"

        if "morpheme" in model_name:
            _is_morpheme = "contains_morphemes"
        else:
            _is_morpheme = "doesnt_contains_morphemes"
        
        for task in data_scores[model_name]:

            if "deft2019" in task or "frenchmedmcqa" in task:
                continue

            if "overall_f1" in data_scores[model_name][task]:
                metric_name = "overall_f1"
            elif "weighted_f1" in data_scores[model_name][task]:
                metric_name = "weighted_f1"
            elif "edrm" in data_scores[model_name][task]:
                metric_name = "edrm"
            else:
                metric_name = "hamming_score"

            task_name = task.replace("|1.0","").replace("ner-fr_","fr-")

            res_map["|".join((_algo, _data, _is_morpheme, task_name ))] = data_scores[model_name][task][metric_name]

    return res_map

def getSubwords():

    res_map = {}

    f_subwords = open("./stats_tokens.json")
    data_subwords = json.load(f_subwords)
    f_subwords.close()

    for task in data_subwords:

        if "deft2019" in task or "frenchmedmcqa" in task:
            continue

        for model_name in data_subwords[task]:

            avg_subwords = sum(data_subwords[task][model_name]) / len(data_subwords[task][model_name])

            model_name = model_name.lower()

            if "sentencepiece" in model_name:
                _algo = "sentencepiece"
            elif "bpe-hf" in model_name:
                _algo = "bpe"

            if "cc100" in model_name:
                _data = "cc100"
            elif "nachos" in model_name:
                _data = "nachos"
            elif "pubmed" in model_name:
                _data = "pubmed"
            elif "wiki" in model_name:
                _data = "wiki"

            if "morpheme" in model_name:
                _is_morpheme = "contains_morphemes"
            else:
                _is_morpheme = "doesnt_contains_morphemes"
            
            corpus_name, task_name = task.lower().split("/")[-1].split("-")
            corpus_name = corpus_name.lower()
            task_name = task_name.lower().replace("_","-").replace("task-1","regr").replace("task-2","cls").replace("french-","ner-")
            
            if "pxcorpus" in corpus_name:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|ner"))] = avg_subwords
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|cls"))] = avg_subwords
            elif "morfitt" in corpus_name:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|cls"))] = avg_subwords
            elif "diamed" in corpus_name:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|cls"))] = avg_subwords
            elif "clister" in corpus_name:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|regr"))] = avg_subwords
            elif "quaero" in corpus_name:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|ner-{task_name}"))] = avg_subwords
            else:
                res_map["|".join((_algo, _data, _is_morpheme, f"{corpus_name}|{task_name}"))] = avg_subwords
    
    return res_map

perfs_map = getPerformances()

with open("./perfs_map.json", 'w') as f:
    json.dump(perfs_map, f, indent=4)

subwords_map = getSubwords()

with open("./subwords_map.json", 'w') as f:
    json.dump(subwords_map, f, indent=4)

common_map = {}

for key in perfs_map:

    common_map[key] = {
        "perf": perfs_map[key],
        "subwords": subwords_map[key],
    }

with open("./common_map.json", 'w') as f:
    json.dump(common_map, f, indent=4)

###################
# AVAILABLE TASKS #
###################
corpus_task = ["|".join(a.split("|")[-2:]) for a in common_map]
corpus_task = list(set(corpus_task))
corpus_task = sorted(corpus_task)

##############
# EACH TASKS #
##############
all_pearsons = []
for _t in corpus_task:

    all_perfs = [common_map[key]["perf"] for key in common_map if _t in key]
    all_subwords_avg = [common_map[key]["subwords"] for key in common_map if _t in key]

    correlation, p_value = pearsonr(all_perfs, all_subwords_avg)
    print(f">> The correlation for {_t} is: {correlation}")
    all_pearsons.append(correlation)

print(f"# Average of all tasks: {sum(all_pearsons)/len(all_pearsons)}")

##############
#  OVERALL   #
##############
all_perfs = [common_map[key]["perf"] for key in common_map]
all_subwords_avg = [common_map[key]["subwords"] for key in common_map]

correlation, p_value = pearsonr(all_subwords_avg, all_perfs)
print("#"*50)
print(f"# The overall correlation is: {correlation}")

import json

import numpy as np
 
f = open("results_sentence_oov.json","r")
data = json.load(f)
f.close()

results = {}

for d in data:

    co = [dd["contains_oov"]["overall_f1"] for dd in data[d]]
    dco = [dd["doesnt_contains_oov"]["overall_f1"] for dd in data[d]]

    results[d] = {
        "contains_oov": sum(co)/len(co),
        "doesnt_contains_oov": sum(dco)/len(dco),
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

with open("./average_results_sentence_oov.json", 'w') as f:
    json.dump(results, f, indent=4, cls=NpEncoder)

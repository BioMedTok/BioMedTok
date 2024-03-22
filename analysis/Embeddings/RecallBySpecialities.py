import json
import nltk
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib  import cm
from random import shuffle
from numpy import linalg as LA
from pqdm.processes import pqdm
import matplotlib.pyplot as plt
from cv2 import estimateAffine2D
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from nltk.cluster import KMeansClusterer
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples

from transformers import CamembertModel, CamembertTokenizer, CamembertConfig, CamembertForTokenClassification, RobertaTokenizerFast, AutoTokenizer

random.seed(42)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

f_in = open("./VocabSpecialites.txt","r")
rows = [a.lower() for a in f_in.read().split("\n")]
f_in.close()
rows = sorted(rows)
terms = [t.split(";")[0] for t in rows]
specialities = [t.split(";")[1] for t in rows]
_uniq_specialities = list(set(specialities))

# print(len(terms))
# print(len(list(set(terms))))
# print(set([x for x in terms if terms.count(x) > 1]))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
TARGET_MODEL = models[0]

# Get embeddings for all sentences
def getBertEmbeddings(tokenizer, model, sentence, cls_token_id=0):
    inputs = tokenizer(sentence, return_tensors='pt').to(device=DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[-1].cpu().detach().numpy().tolist()
    return hs[0][cls_token_id]

# Fetch and store the model in the cache
def getModel(MODEL_NAME):

    if "sentencepiece" in MODEL_NAME.lower():
        tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    elif "bpe-hf" in MODEL_NAME.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    config = CamembertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = CamembertForTokenClassification.from_pretrained(MODEL_NAME, config=config).to(DEVICE)
    
    return tokenizer, model

all_embeddings = {}
all_projected_embeddings = {}

for MODEL_NAME in models:

    ####################
    # MODELS CACHING   #
    ####################
    tokenizer, model = getModel(MODEL_NAME)

    ##############
    # EMBEDDINGS #
    ##############
    all_embeddings[MODEL_NAME] = {term: getBertEmbeddings(tokenizer, model, term) for term in terms}

    del tokenizer
    del model

MODE = "disable"
# MODE = "disable+normalized"
# MODE = "enable"
# MODE = "enable+normalized"

if "enable" in MODE:

    all_projected_embeddings[TARGET_MODEL] = list(all_embeddings[TARGET_MODEL].values())

    for MODEL_NAME in models[1:]:

        in_arr = torch.Tensor([a for a in list(all_embeddings[MODEL_NAME].values())])
        in_target_arr = torch.Tensor([a for a in list(all_embeddings[TARGET_MODEL].values())])

        _weights, res, r1, s = np.linalg.lstsq(in_arr, in_target_arr)

        all_projected_embeddings[MODEL_NAME] = []

        for og_vec, _w in zip(in_arr, _weights):

            _r  = torch.from_numpy(_weights).matmul(og_vec)

            if "+normalized" in MODE:
                _r = _r.divide(LA.norm(_r))

            all_projected_embeddings[MODEL_NAME].append(_r.tolist())

    all_embeddings[MODEL_NAME] = {_term: _emb for _term, _emb in zip(terms, all_projected_embeddings[MODEL_NAME])}

def getDistance(_m1_emb, _m2_emb):

    return abs(float(cosine_similarity(
        np.array(_m1_emb).reshape(1, -1),
        np.array(_m2_emb).reshape(1, -1)
    )[0][0]))


def getAllDistances(_m1_emb, model_embs):

    res = []

    for _m_emb in model_embs:

        _dist = getDistance(_m1_emb, _m_emb)
        res.append(_dist)
    
    return res

def compute_all_dst(_t, _emb_1, _all_emb_2, _terms):
    return (
        _t,
        [(_tt, _d) for _tt, _d in zip(_terms, getAllDistances(_emb_1, _all_emb_2))]
    )

def getRecallAtK(model_1_emb, model_2_emb, k):

    model_1_emb = {a: b for a, b in model_1_emb}

    # print(model_1 + " - " + model_2)

    # print("Get distances:")

    result = []
    for _t in terms:
        _r = compute_all_dst(_t, model_1_emb[_t], model_2_emb, terms)
        result.append(_r)

    # result = pqdm([[
    #     _t, model_1_emb[_t], model_2_emb, terms
    # ] for _t in terms], compute_all_dst, n_jobs=12, argument_type='args')
    # print(result)

    _local_matrix = {_t: {_r[0]: _r[1] for _r in _res} for _t, _res in result}
    # print(_local_matrix.keys())
    # print("_local_matrix")

    # _local_matrix = {_t: {_tt: getDistance(all_embeddings[model_1][_t], all_embeddings[model_2][_tt]) for _tt in terms} for _t in tqdm(terms)}

    res_positions = {_t: -1 for _t in terms}

    # print("Sort and get positions:")

    for _t in tqdm(terms):
        
        _sorted = sorted(_local_matrix[_t].items(), key=lambda x: (-x[1], x[0]))
        # print(_sorted)
        # print("_sorted")

        positions_list = [_s[0] for _s in _sorted]
        # print(positions_list)
        # print("len(_local_matrix[_t].keys())")
        # print(len(_local_matrix[_t].keys()))
        # print("positions_list")
        # print(len(positions_list))
        # print(len(terms))
        # print(len(all_embeddings[model_2].values()))

        position_term = positions_list.index(_t)
        res_positions[_t] = position_term+1

    _map_k = []

    for _pos in res_positions.values():

        if _pos <= k:
            _current_k = 1 / _pos
        else:
            _current_k = 0

        _map_k.append(_current_k)

    _map_k = (1 / len(_map_k)) * sum(_map_k)

    return _map_k

###############################################
# DISTANCE FOR EACH TERM OF THE SPEICIALITIES #
###############################################
distances_terms_matrix_sorted = {}

def EDIT(text):
    return text.replace("BioMedTok/","").replace("-FR","").replace("-Morphemes","+M").replace("BPE-HF","BPE").replace("SentencePieceBPE","SPM")

_table_out = []

_table_out.append(
    " & " + " & ".join([EDIT(m) for m in models]) + " \\\\ \\hline"
)

for _m1 in models:

    _elements = []

    for _m2 in models:

        _elements.append([
            list(all_embeddings[_m1].items()),
            list(all_embeddings[_m2].values()),
            10
        ])

    _all_recall = pqdm(_elements, getRecallAtK, n_jobs=12, argument_type='args')

    _all_recall = ["{:.2f}".format(_ar*100) for _ar in _all_recall]
    
    _line = EDIT(_m1) + " & " + " & ".join(_all_recall) + " \\\\ \\hline"
    _table_out.append(_line)

_table_str = "\n".join(_table_out)
print(_table_str)

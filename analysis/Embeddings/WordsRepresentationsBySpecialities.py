
import json
import nltk
import torch
import random
import numpy as np
import pandas as pd
from matplotlib  import cm
from random import shuffle
from numpy import linalg as LA
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

MAX_ELEMENTS = 20

f_in = open("./VocabSpecialites.txt","r")
rows = [a.lower() for a in f_in.read().split("\n")]
f_in.close()
rows = sorted(rows)
terms = [t.split(";")[0] for t in rows]
specialities = [t.split(";")[1] for t in rows]
_uniq_specialities = list(set(specialities))

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

MODELS_MARKERS = ["o" , "d" , "h", "s"]

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

for MODEL_NAME in models:

    ####################
    # MODELS CACHING   #
    ####################
    tokenizer, model = getModel(MODEL_NAME)

    ##############
    # EMBEDDINGS #
    ##############
    all_embeddings[MODEL_NAME] = {term: getBertEmbeddings(tokenizer, model, term) for term in terms}

###############################################
# DISTANCE FOR EACH TERM OF THE SPEICIALITIES #
###############################################
distances = {}
for _m in models[1:]:
    for _speciality in _uniq_specialities:
        distances[(_m, _speciality)] = {_t: 0.0 for _i, _t in enumerate(terms) if specialities[_i] == _speciality} 

print("Distances:")
print(distances.keys())

for _term, _spe in zip(terms, specialities):

    target_embedding = all_embeddings[TARGET_MODEL][_term]

    for MODEL_NAME in models[1:]:

        _target = np.array(target_embedding).reshape(1, -1)
        _current = np.array(all_embeddings[MODEL_NAME][_term]).reshape(1, -1)

        _distance = float(cosine_similarity(
            _target,
            _current
        )[0][0])

        distances[(MODEL_NAME, _spe)][_term] = _distance

##################################
# AVG DISTANCE PER SPEICIALITIES #
##################################
avg_distance_per_specialities = {(_m, _speciality): 0.0 for _m, _speciality in zip(models, _uniq_specialities)}

for MODEL_NAME in models[1:]:
        
    for _spe in _uniq_specialities:

        _avg_spe = sum(distances[(MODEL_NAME, _spe)].values()) / len(distances[(MODEL_NAME, _spe)].values())

        avg_distance_per_specialities[(MODEL_NAME, _spe)] = abs(_avg_spe)

###############
# BUILD TABLE #
###############

_table = []

_table.append(
    f"Models & " + " & ".join([_spe.capitalize() for _spe in _uniq_specialities]) + " \\\\ \\hline"
)

all_values = []

for MODEL_NAME in models[1:]:
    
    _line = []

    for _spe in _uniq_specialities:
        _value = avg_distance_per_specialities[(MODEL_NAME, _spe)]*100
        all_values.append(_value)
        _line.append("{:.7f}".format(_value))
        
    # _line = " & ".join(["{:.2f}".format(avg_distance_per_specialities[(MODEL_NAME, _spe)]) for _spe in specialities])
    _line = " & ".join(_line)
    _line = f"{MODEL_NAME.replace('BioMedTok/','')} & {_line} \\\\ \\hline"
    # _line = f"{MODEL_NAME.replace('BioMedTok/','')} / {TARGET_MODEL.replace('BioMedTok/','')} & {_line} \\\\ \\hline"
    _table.append(_line)

_table = "\n".join(_table)

print(_table)
print(max(all_values))
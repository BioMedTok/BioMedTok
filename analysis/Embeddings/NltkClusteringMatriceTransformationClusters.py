
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
from sklearn.metrics import silhouette_score, silhouette_samples

from transformers import CamembertModel, CamembertTokenizer, CamembertConfig, CamembertForTokenClassification, RobertaTokenizerFast, AutoTokenizer

random.seed(42)

MAX_ELEMENTS = 20

# f_in = open("VocabMedicalSpider.txt","r")
# terms = f_in.read().split("\n")
# f_in.close()

f_in = open("./VocabSpecialites.txt","r")
rows = [a.lower() for a in f_in.read().split("\n")]
rows = sorted(rows)
terms = [t.split(";")[0] for t in rows]
classes = [t.split(";")[1] for t in rows]
_classes = sorted(list(set(classes)))
classes = [_classes.index(c) for c in classes]
colors = ['red','green','blue','cyan','black', 'yellow']
colors = cm.rainbow(np.linspace(0, 1, len(colors)))
color_classes = [colors[c] for c in classes]
f_in.close()

f_out_classes = open("./clustering_outputs_clusters/classes.txt","w")
f_out_classes.write("\n".join(_classes))
f_out_classes.close()

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

# CONFIGS_EXPS = [
#     {
#         "name_run": "SentencePiece NACHOS With vs Without morphemes",
#         "models": [
#             "BioMedTok/SentencePieceBPE-NACHOS-FR",
#             "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
#         ]
#     },
#     {
#         "name_run": "SentencePiece 4 four sources ( NACHOS - PubMed - CC100 - Wiki )",
#         "models": [
#             "BioMedTok/SentencePieceBPE-NACHOS-FR",
#             "BioMedTok/SentencePieceBPE-PubMed-FR",
#             "BioMedTok/SentencePieceBPE-CC100-FR",
#             "BioMedTok/SentencePieceBPE-Wikipedia-FR",
#         ]
#     },
#     {
#         "name_run": "NACHOS With vs Without & BPE vs SentencePiece",
#         "models": [
#             "BioMedTok/BPE-HF-NACHOS-FR",
#             "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",
#             "BioMedTok/SentencePieceBPE-NACHOS-FR",
#             "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
#         ]
#     },
# ]

# models_displayed = [

#     "BioMedTok/BPE-HF-NACHOS-FR",
#     "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",

#     # "BioMedTok/BPE-HF-PubMed-FR",
#     # "BioMedTok/BPE-HF-PubMed-FR-Morphemes",
    
#     # "BioMedTok/BPE-HF-CC100-FR",
#     # "BioMedTok/BPE-HF-CC100-FR-Morphemes",
    
#     # "BioMedTok/BPE-HF-Wikipedia-FR",
#     # "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes",
    
#     # "BioMedTok/SentencePieceBPE-NACHOS-FR",
#     # "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
    
#     # "BioMedTok/SentencePieceBPE-PubMed-FR",
#     # "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes",
    
#     # "BioMedTok/SentencePieceBPE-CC100-FR",
#     # "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes",
    
#     # "BioMedTok/SentencePieceBPE-Wikipedia-FR",
#     # "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes",
# ]

MODELS_MARKERS = ["o" , "d" , "h", "s"]
# MODELS_MARKERS = ["." , "," , "o" , "v" , "^" , "<", ">"]

# Target vector space
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

def distance_from_centroid(row):
    return distance_matrix([row['emb']], [row['centroid'].tolist()])[0][0]

# all_models = {}
all_embeddings = {}
all_projected_embeddings = {}
MODELS_TSNE = {}

for MODEL_NAME in models:

    ####################
    # CACHE FOR MODELS #
    ####################

    tokenizer, model = getModel(MODEL_NAME)

    # all_models[MODEL_NAME] = {
    #     "tokenizer": tokenizer,
    #     "model": model,
    # }

    ##############
    # EMBEDDINGS #
    ##############
    all_embeddings[MODEL_NAME] = [getBertEmbeddings(tokenizer, model, term) for term in terms]

################################################################################
# LEAST SQUARE MATRIX TRANSFORMATION OF EMBEDDINGS                             #
# > Excluding the first model which is considered as the target representation #
################################################################################

all_projected_embeddings[TARGET_MODEL] = all_embeddings[TARGET_MODEL]

# MODE = "disable"
# MODE = "disable+normalized"
# MODE = "enable"
MODE = "enable+normalized"

if "enable" in MODE:

    for MODEL_NAME in models[1:]:

        in_arr = torch.Tensor([a for a in all_embeddings[MODEL_NAME]])
        in_target_arr = torch.Tensor([a for a in all_embeddings[TARGET_MODEL]])

        _weights, res, r1, s = np.linalg.lstsq(in_arr, in_target_arr)

        all_projected_embeddings[MODEL_NAME] = []

        for og_vec, _w in zip(in_arr, _weights):

            _r  = torch.from_numpy(_weights).matmul(og_vec)

            if "+normalized" in MODE:
                _r = _r.divide(LA.norm(_r))

            all_projected_embeddings[MODEL_NAME].append(_r.tolist())

#########
# T-SNE #
#########

MODE_TSNE = "t-sne all 1D"
# MODE_TSNE = "t-sne all"
# MODE_TSNE = "t-sne individual"
# MODE_TSNE = "PCA all"
# MODE_TSNE = "pca"
# MODE_TSNE = "PCA all 3D"
# MODE_TSNE = "t-sne individual 3D"
# MODE_TSNE = "t-sne all 3D"
# MODE_TSNE = "PCA 3D"
# MODE_TSNE = "nothing"

if MODE_TSNE == "t-sne all":

    elements_indexes = {_m: [] for _m in models}

    ALL_VECTORS = []

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # For each vectors
        for _v in _VECTORS:

            # Add it to the general list
            ALL_VECTORS.append(_v)

            # Save the index of its embeddings into the models dictionnary
            elements_indexes[MODEL_NAME].append(len(ALL_VECTORS) - 1)

    # Apply the T-SNE on all the embeddings from all the models
    ALL_X_TSNE = TSNE(n_components=2).fit_transform(np.array(ALL_VECTORS))

    # Then for each of the models
    for MODEL_NAME in models:

        # Get the corresponding embedding after T-SNE nromalization based on the indexes affected to it
        MODELS_TSNE[MODEL_NAME] = [ALL_X_TSNE[idx] for idx in elements_indexes[MODEL_NAME]]

elif MODE_TSNE == "t-sne all 1D":

    elements_indexes = {_m: [] for _m in models}

    ALL_VECTORS = []

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # For each vectors
        for _v in _VECTORS:

            # Add it to the general list
            ALL_VECTORS.append(_v)

            # Save the index of its embeddings into the models dictionnary
            elements_indexes[MODEL_NAME].append(len(ALL_VECTORS) - 1)

    # Apply the T-SNE on all the embeddings from all the models
    ALL_X_TSNE = TSNE(n_components=1).fit_transform(np.array(ALL_VECTORS))

    # Then for each of the models
    for MODEL_NAME in models:

        # Get the corresponding embedding after T-SNE nromalization based on the indexes affected to it
        MODELS_TSNE[MODEL_NAME] = [ALL_X_TSNE[idx] for idx in elements_indexes[MODEL_NAME]]

elif MODE_TSNE == "t-sne all 3D":

    elements_indexes = {_m: [] for _m in models}

    ALL_VECTORS = []

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # For each vectors
        for _v in _VECTORS:

            # Add it to the general list
            ALL_VECTORS.append(_v)

            # Save the index of its embeddings into the models dictionnary
            elements_indexes[MODEL_NAME].append(len(ALL_VECTORS) - 1)

    # Apply the T-SNE on all the embeddings from all the models
    ALL_X_TSNE = TSNE(n_components=3).fit_transform(np.array(ALL_VECTORS))

    # Then for each of the models
    for MODEL_NAME in models:

        # Get the corresponding embedding after T-SNE nromalization based on the indexes affected to it
        MODELS_TSNE[MODEL_NAME] = [ALL_X_TSNE[idx] for idx in elements_indexes[MODEL_NAME]]

elif MODE_TSNE == "PCA all":

    elements_indexes = {_m: [] for _m in models}

    ALL_VECTORS = []

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # For each vectors
        for _v in _VECTORS:

            # Add it to the general list
            ALL_VECTORS.append(_v)

            # Save the index of its embeddings into the models dictionnary
            elements_indexes[MODEL_NAME].append(len(ALL_VECTORS) - 1)

    # Apply the T-SNE on all the embeddings from all the models
    ALL_X_PCA = PCA(n_components=2).fit_transform(np.array(ALL_VECTORS))

    # Then for each of the models
    for MODEL_NAME in models:

        # Get the corresponding embedding after T-SNE nromalization based on the indexes affected to it
        MODELS_TSNE[MODEL_NAME] = [ALL_X_PCA[idx] for idx in elements_indexes[MODEL_NAME]]

elif MODE_TSNE == "PCA all 3D":

    elements_indexes = {_m: [] for _m in models}

    ALL_VECTORS = []

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # For each vectors
        for _v in _VECTORS:

            # Add it to the general list
            ALL_VECTORS.append(_v)

            # Save the index of its embeddings into the models dictionnary
            elements_indexes[MODEL_NAME].append(len(ALL_VECTORS) - 1)

    # Apply the T-SNE on all the embeddings from all the models
    ALL_X_PCA = PCA(n_components=3).fit_transform(np.array(ALL_VECTORS))

    # Then for each of the models
    for MODEL_NAME in models:

        # Get the corresponding embedding after T-SNE nromalization based on the indexes affected to it
        MODELS_TSNE[MODEL_NAME] = [ALL_X_PCA[idx] for idx in elements_indexes[MODEL_NAME]]

elif MODE_TSNE == "t-sne individual":

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # Apply the T-SNE on the embeddings from one model at the time
        MODELS_TSNE[MODEL_NAME] = TSNE(n_components=2).fit_transform(np.array(_VECTORS))

elif MODE_TSNE == "t-sne individual 3D":

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        # Apply the T-SNE on the embeddings from one model at the time
        MODELS_TSNE[MODEL_NAME] = TSNE(n_components=3).fit_transform(np.array(_VECTORS))

elif MODE_TSNE == "PCA 3D":

    print("PCA Individual 3D")

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        MODELS_TSNE[MODEL_NAME] = PCA(n_components=3).fit_transform(np.array(_VECTORS))

elif MODE_TSNE == "pca":

    print("PCA Individual")

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        MODELS_TSNE[MODEL_NAME] = PCA(n_components=2).fit_transform(np.array(_VECTORS))

else:

    print("Nothing full size")

    for MODEL_NAME in models:

        if MODE == "enable":
            _VECTORS = all_projected_embeddings[MODEL_NAME]
        else:
            _VECTORS = all_embeddings[MODEL_NAME]

        MODELS_TSNE[MODEL_NAME] = _VECTORS

common_x = []
common_y = []
common_z = []
common_m = []
common_terms = []

avg_cluster_dist = {}

################################
# RUN A ENTIRE EXPERIMENTATION #
################################
for idx, MODEL_NAME in enumerate(models):

    print(MODEL_NAME)

    # ###############
    # # DATA FIGURE #
    # ###############
    # _x = [e[0] for e in MODELS_TSNE[MODEL_NAME]]
    # _y = [e[1] for e in MODELS_TSNE[MODEL_NAME]]
    # _z = color_classes
    # # _z = data['cluster'].values.tolist()
    
    # common_x.extend(_x)
    # common_y.extend(_y)
    # common_z.extend(_z)
    # common_terms.extend(terms)
    # # common_m.extend([MODELS_MARKERS[idx]]*len(terms))

    ##############
    # CLUSTERING #
    ##############
    data = [[t, te] for t, te in zip(terms, MODELS_TSNE[MODEL_NAME])]
    data = pd.DataFrame(data, columns=['text', 'emb'])
    # print(data)

    metric_dbscan_avg = [] 
    metric_dbscan_avg_ari = [] 

    metric_kmeans_avg = [] 
    metric_kmeans_avg_ari = [] 
    
    for _k in range(2, len(_classes)+5):

        kmeans = KMeans(n_clusters=_k, n_init="auto")
        kmeans.fit(MODELS_TSNE[MODEL_NAME])

        _metric_kmeans = silhouette_score(data["emb"].tolist(), kmeans.labels_)
        _metric_kmeans_ari= adjusted_rand_score(classes, kmeans.labels_)

        metric_kmeans_avg.append(_metric_kmeans)
        metric_kmeans_avg_ari.append(_metric_kmeans_ari)

        print("The metric SS KMeans value is: ", _metric_kmeans, " for k=", _k)
        print("The metric ARI KMeans value is: ", _metric_kmeans_ari, " for k=", _k)
        print("The metric SSE KMeans value is: ", kmeans.inertia_, " for k=", _k)

    # avg_ss = round(sum(metric_avg) / len(metric_avg), 2)
    # print("Average SS values on kmeans is: ", avg_ss)

    # avg_ari = round(sum(metric_avg_ari) / len(metric_avg_ari), 2)
    # print("Average ARI values on kmeans is: ", avg_ari)

    print("*"*50)




    # NUM_CLUSTERS = len(_classes)
    # print(NUM_CLUSTERS)

    # kclusterer = KMeansClusterer(
    #     NUM_CLUSTERS,
    #     distance=nltk.cluster.util.cosine_distance,
    #     repeats=25,
    #     avoid_empty_clusters=True,
    # )
    # print("kclusterer")
    # print(kclusterer)

    # assigned_clusters = kclusterer.cluster(data["emb"], assign_clusters=True)
    # print("assigned_clusters")
    # print(assigned_clusters)

    # data['classes'] = pd.Series([_classes[_c] for _c in classes], index=data.index)

    # data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    # data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

    # data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)

    # data = data.drop('centroid', axis=1)
    # data = data.drop('emb', axis=1)
    # # data = data.drop('emb_tsne', axis=1)
    # print(data)

    # data = data.reset_index(drop=True)
    
    # data.to_csv(f"clustering_outputs_clusters/clusters_{MODEL_NAME.replace('/','_')}.csv")








    # clusters_distances = {}

    # for index, row in data.iterrows():

    #     if row["cluster"] not in clusters_distances:
    #         clusters_distances[row["cluster"]] = []

    #     clusters_distances[row["cluster"]].append(row["distance_from_centroid"])
    
    # for _cls in clusters_distances:
        
    #     if MODEL_NAME.replace('/','_') not in avg_cluster_dist:
    #         avg_cluster_dist[MODEL_NAME.replace('/','_')] = {}

    #     avg_cluster_dist[MODEL_NAME.replace('/','_')][_classes[_cls]] = sum(clusters_distances[_cls]) / len(clusters_distances[_cls])

# with open("./clustering_outputs_clusters/avg_cluster_dist.json", 'w') as f:
#     json.dump(avg_cluster_dist, f, indent=4)


# print(_classes)
# print("#"*50)
# for _model in avg_cluster_dist:

#     _out = "    ".join(["{:.3f}".format(avg_cluster_dist[_model][_cls]) for _cls in _classes])
#     # print(_model)
#     print(_out)
#     # print()

# ###############################
# # DISPLAY COMMON VECTOR SPACE #
# ###############################
# for l_x, l_y, l_z, l_m in zip(common_x, common_y, common_z, common_m):
#     plt.scatter(l_x, l_y, s=1, c=l_z, cmap=cm.jet, marker=l_m)
#     # plt.scatter(l_x, l_y, s=1, marker=l_m, c="r")
#     # plt.scatter(l_x, l_y, s=1, marker=l_m, facecolors='none', edgecolors=l_z, c=l_z)
#     # plt.scatter(l_x, l_y, s=20, color=l_z, marker=l_m, cmap=cm.jet)

# for c1, c2, t in zip(common_x, common_y, common_terms):
#     # 'weight': 'bold'
#     plt.text(c1, c2, t, fontdict={'family': 'serif', 'color':  'black', 'size': 2})

# plt.savefig(f"clustering_outputs_clusters/{MODEL_NAME.replace('/','_')}.png", bbox_inches='tight', dpi=600)
# plt.clf()


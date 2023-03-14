"""
Utility Functions, Evaluation
"""

import json
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.metrics import f1_score
from scipy.spatial import distance
from nltk.stem.snowball import EnglishStemmer
import gensim.downloader as gensim_api
import torch


### config files
def load_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return dict()
        
def check_config(config, key):
    return (key not in config) or (not config[key])

def write_config(config: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(config, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        return pk.load(f)

def write_pickle(path, content):
    with open(path, 'wb') as f:
        pk.dump(content, f, protocol=4)


### Metrics
def accuracy(label, pred):
    return (label == pred).mean()

def macro_f1(label, pred):
    return f1_score(label, pred, average='macro')

def micro_f1(label, pred):
    return f1_score(label, pred, average='micro')

### Similarity
def cosine_similarity(u, v):
    return 1 - distance.cosine(u, v)

def w2v_cosine_similiarity(model, u, v):
    stemmer = EnglishStemmer()
    if u not in model:
        stem_u = stemmer.stem(u)
        if stem_u not in model:
            vec_u = composite_w2v_embedding(model, u)
            if vec_u is None:
                print(f'{u} not in vocabulary')
                return
        else:
            vec_u = model[stem_u]
    else:
        vec_u = model[u]
    
    if v not in model:
        stem_v = stemmer.stem(v)
        if stem_v not in model:
            vec_v = composite_w2v_embedding(model, v)
            if vec_v is None:
                print(f'{v} not in vocabulary')
                return
        else:
            vec_v = model[stem_v]
    else:
        vec_v = model[v]
    
    return cosine_similarity(vec_u, vec_v)

def composite_w2v_embedding(model, word):
    stemmer = EnglishStemmer().stem
    total = []
    for piece in word.split('_'):
        if piece not in model:
            piece = stemmer(piece)
        if piece in model:
            total.append(model[piece])
    if not total:
        return
    return np.mean(np.array(total), axis=0)

### load pre-trained Word2Vec
def load_w2v_model():
    """ 
    Load Word2Vec Pre-Trained Vectors
    """
    wv_from_bin = gensim_api.load("word2vec-google-news-300")

    print("Loaded vocab size %i" % len(wv_from_bin.key_to_index))
    return wv_from_bin


### plotting
def save_plot(fig, path):
    fig.savefig(path, transparent=True)


# torch utils
def tensor_to_numpy(tensor):
    if tensor.device.type == 'cuda':
        return tensor.clone().detach().cpu().numpy()
    else:
        return tensor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

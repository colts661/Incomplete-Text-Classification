"""
Utility Functions, Evaluation
"""

from data import *
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import nltk
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet
from nltk.stem.snowball import EnglishStemmer
import gensim.downloader as gensim_api


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

def wordnet_similarity(u, v):
    """
    Use Wu-Palmer similarity from WordNet for granuality check
    """
    return wordnet.synsets(u)[0].wup_similarity(wordnet.synsets(v)[0])

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
def display_vector_2d(reps, centroids=None, pca_dim=100, tsne_perp=30):
    """
    Displays the vector representations in 2D
    """
    # reduce dimension with PCA
    pca = PCA(n_components=pca_dim)
    pca_rep = pca.fit_transform(reps)

    # t-SNE visualization transformation
    tsne = TSNE(n_iter=5000, init='pca', learning_rate='auto', perplexity=tsne_perp)
    tsne_rep = tsne.fit_transform(pca_rep)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))        
    ax.scatter(tsne_rep[:, 0], tsne_rep[:, 1], edgecolors='k', c='gray', s=2)
    if centroids is not None:
        pca_centroids = pca.transform(centroids)
        tsne_centroids = tsne.transform(pca_centroids)
        ax.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1], edgecolors='k', c='red', s=30)
    ax.set_title('Unconfident Document Representations in 2D')
    return fig


def save_plot(fig, path):
    fig.savefig(path, transparent=True)
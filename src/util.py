"""
Utility Functions for Replication
"""

from data import *
import json
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from scipy.spatial import distance


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet
from nltk.stem.snowball import EnglishStemmer
import gensim.downloader as gensim_api


### config files
def load_config(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return dict()
        
def check_config(config, key):
    return (key not in config) or (not config[key])


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
    stemmer = EnglishStemmer().stem
    if u not in model:
        stem_u = stemmer.stem(u)
        if stem_u not in model:
            print(f'{u} not in vocabulary')
            return
        else:
            u = stem_u
    
    if v not in model:
        stem_v = stemmer.stem(v)
        if stem_v not in model:
            print(f'{v} not in vocabulary')
            return
        else:
            v = stem_v
    
    return cosine_similarity(model[u], model[v])


### load pre-trained Word2Vec
def load_w2v_model():
    """ 
    Load Word2Vec Pre-Trained Vectors
    """
    wv_from_bin = gensim_api.load("word2vec-google-news-300")

    print("Loaded vocab size %i" % len(wv_from_bin.key_to_index))
    return wv_from_bin


### MAIN EVALUATION METHOD
def evaluate(
    data: Data, new_class_pred, 
    similarity_func=None, w2v_model=None,
    return_sim=False, return_table=False
):
    """
    Evaluate the model based on similarity mapping and macro-F1 score.
    Assuming all suggested classes can be mapped to removed labels.
    
    ---
    Parameters:
               data: Data class object
     new_class_pred: predicted labels of non-existing labels
    similarity_func: function for similarity score between 2 words,
                     default cosine Word2Vec similarity
         return_sim: whether to return mapped similarity, default False
       return_table: whether to return label and prediction, default False
    """
    assert len(data.unlabeled_labels) == len(new_class_pred)
    
    if similarity_func is None:
        if w2v_model is None:
            model = gensim_api.load("word2vec-google-news-300")
        else:
            model = w2v_model
        similarity_func = lambda u, v: w2v_cosine_similiarity(model, u, v)
    
    # existing classes evaluation
    existing_classes = set(data.labeled_labels)
    existing_idx, existing_pred, existing_label = zip(*[
        (idx, pred, truth) 
        for idx, (pred, truth) in enumerate(zip(new_class_pred, data.unlabeled_labels))
        if pred in existing_classes
    ])
    
    # new class
    new_classes = set(data.unlabeled_labels) - set(data.labeled_labels)
    pending_idx, pending_pred, pending_label = zip(*[
        (idx, pred, truth) 
        for idx, (pred, truth) in enumerate(zip(new_class_pred, data.unlabeled_labels))
        if pred not in existing_classes
    ])
    
    # unconstrained mapping
    suggested_classes = set(pending_pred)
    similarity = pd.DataFrame([
        [similarity_func(p, t) for t in new_classes] 
        for p in suggested_classes
    ], index=suggested_classes, columns=new_classes)

    mapping = similarity.idxmax(axis=1).to_dict()
    
    # combine and evaluate
    full_idx = list(existing_idx) + list(pending_idx)
    full_pred = list(existing_pred) + list(pending_pred)
    full_map = list(existing_pred) + [mapping[pred] for pred in pending_pred]
    full_label = list(existing_label) + list(pending_label)
    full_table = pd.DataFrame(
        {'prediction': full_pred, 'mapping': full_map, 'label': full_label}, 
        index=full_idx
    ).sort_index()
    score = macro_f1(full_map, full_label)
    
    if (not return_sim) and (not return_table):
        return score
    else:
        return tuple(
            [score] + 
            ([similarity] if return_sim else []) + 
            ([full_table] if return_table else [])
        )
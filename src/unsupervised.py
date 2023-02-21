# basic packages
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project dependencies
from data import *
from util import *

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError


class Clustering_Model:
    def __init__(self, method, vectors, vector_idx):
        supports = ['kmeans', 'gmm']
        assert method.lower() in supports, f"Only supports {', '.join(supports)}"
        self.method = method
        self.vectors = vectors
        self.vector_idx = vector_idx

    def fit_transform(self, n_classes=10):
        # define model
        if self.method.lower() == 'kmeans':
            self.model = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
        elif self.method.lower() == 'gmm':
            self.model = GaussianMixture(n_components=n_classes, n_init=10, random_state=0)

        # run clustering
        self.cluster_results = self.model.fit_predict(self.vectors)
        return self.cluster_results
"""
A collection of unsupervised learning modules: dimensionality reduction, clustering
"""

# basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Clustering_Model:
    def __init__(self, method, vectors, vector_idx, docs):
        supports = ['kmeans', 'gmm']
        assert method.lower() in supports, f"Only supports {', '.join(supports)}"
        self.method = method
        self.vectors = vectors
        self.vector_idx = vector_idx
        self.docs = docs

    def fit_transform(self, n_classes=10):
        # define model
        if self.method.lower() == 'kmeans':
            self.model = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
        elif self.method.lower() == 'gmm':
            self.model = GaussianMixture(n_components=n_classes, n_init=10, random_state=0)

        # run clustering
        self.cluster_results = pd.Series(self.model.fit_predict(self.vectors), index=self.vector_idx)
        self.cluster_prob = pd.DataFrame(self.model.predict_proba(self.vectors), index=self.vector_idx)
        return self.cluster_results
    
    def pick_sample_for_label(self, k=25):
        """
        Pick k most confident examples for label generation
        """
        assert hasattr(self, "model"), "Please fit the model first"
        
        # select for each class
        all_samples = dict()
        for cluster_id in range(self.cluster_prob.shape[1]):
            one_cluster = self.cluster_prob[self.cluster_results == cluster_id][cluster_id]
            
            # first attempt: pick by probability
            sampled = one_cluster.nlargest(k)
            
            # second attempt: same probability
            if sampled.iloc[0] == sampled.iloc[-1]:
                sampled = one_cluster[one_cluster == sampled.iloc[0]]
                sampled = sampled.sample(k, replace=False, random_state=42)
        
            all_samples[cluster_id] = sampled.index.to_numpy()
        
        return all_samples
    
    def get_df_new_classes(self):
        return pd.DataFrame({'sentence': [
            ' '.join(doc) for doc in self.docs.values()
        ], 'label': self.cluster_results}, index=self.vector_idx)


class Dimensionality_Reduction:
    def __init__(self, method):
        supports = ['pca', 'tsne']
        assert method.lower() in supports, f"Only supports {', '.join(supports)}"
        self.method = method

    def fit_transform(self, rep, **config):
        # define model
        if self.method.lower() == 'pca':
            self.dimension = config['dimension']
            self.model = PCA(n_components=self.dimension, random_state=42)
        elif self.method.lower() == 'tsne':
            self.dimension = config['dimension']
            self.perplexity = config['perplexity']
            self.model = TSNE(
                n_components=self.dimension, 
                n_iter=5000, init='pca', learning_rate='auto', 
                perplexity=self.perplexity
            )
        
        # fit
        return self.model.fit_transform(rep)

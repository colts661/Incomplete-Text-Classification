"""
A collection of models
"""

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

# NLP packages
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# misc
from tqdm import tqdm


class Tfidf_Model:
    def fit_transform(self, corpus: pd.DataFrame):
        self.model = TfidfVectorizer(stop_words='english')
        tfidf_values = self.model.fit_transform(corpus['sentence'])
        self.values_matrix = pd.DataFrame(
            tfidf_values.toarray(), 
            columns=self.model.get_feature_names_out(),
            index=corpus.index
        )
    
    def get_top_dict(self, stop_words=None, k=10, strict=False, keep_values=False):
        if not hasattr(self, 'model'):
            raise NotImplementedError('Please fit the model first')
            
        if stop_words is None:
            stop_words = ENGLISH_STOP_WORDS
            
        # get top values, duplicate possible
        top_dict = {}
        all_top = []
        if strict:
            top_take = k
        else:
            top_take = max(10, k * 3)
        
        for label, ser in self.values_matrix.iterrows():
            if keep_values:
                top = [(word, value) for word, value in (
                    ser[~ser.index.isin(stop_words)].nlargest(top_take)
                    .to_dict().items()
                )]
                top_dict[label] = top
                all_top.extend(map(lambda tup: tup[0], top))
            else:
                top = ser[~ser.index.isin(stop_words)].nlargest(top_take).index.tolist()
                top_dict[label] = top
                all_top.extend(top)
        
        # remove duplicates
        duplicates = {word for word, count in Counter(all_top).most_common() if count > 1}
        duplicates = duplicates.union(stop_words)

        for label, top_words in top_dict.items():
            if keep_values:
                top_dict[label] = [tup for tup in top_words if tup[0] not in duplicates][:k]
            else:
                top_dict[label] = [word for word in top_words if word not in duplicates][:k]

        return top_dict


class Word2Vec_Model:
    """
    Baseline model that utilizes the Word2Vec algorithm to learn
    vector representations for words.
    """

    def __init__(self, corpus, labels=None, seedwords=None):
        self.corpus = corpus
        
        if labels and seedwords is None:  # seed word set to class name only
            self.labels = labels
            self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])
            self.seedwords = {s: [s] for s in self.labels}
        if seedwords and labels is None:
            self.seedwords = seedwords
            self.labels = list(self.seedwords)
            self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])            
    
    def fit(self, **config):
        """
        Train the Word2Vec model using config in model.
        """
        self.config = config
        self.model = Word2Vec(self.corpus, **config, callbacks=[Word2Vec_Callback()])
    
    def load_model(self, path_or_model):
        if os.path.exists(path_or_model):
            assert '.model' in path_or_model
            self.model = Word2Vec.load(path_or_model)
            self.config = load_config(path_or_model.replace('.model', '_config.json'))
        elif isinstance(path_or_model, str):
            try:
                self.model = gensim_api.load(path_or_model)
            except ValueError:
                pass
        else:
            raise FileNotFoundError('Model or path to model not found')
    
    def predict(self):
        self.check_fitted()

        # find representations
        self.doc_rep = np.empty((len(self.corpus), self.model.vector_size))
        self.label_rep = np.empty((len(self.labels), self.model.vector_size))

        for i, doc in tqdm(
            enumerate(self.corpus), 
            "Finding Document Representations"
        ):
            embed = self.get_avg_embedding(doc)
            if embed is not None:
                self.doc_rep[i] = embed
    
        for i, seeds in tqdm(
            enumerate(self.seedwords.values()), 
            "Finding Label Representations"
        ):
            embed = self.get_avg_embedding(seeds)
            if embed is not None:
                self.label_rep[i] = embed
        
        # find relevance
        self.relevance = np.empty((len(self.corpus), len(self.labels)))

        for i, doc in tqdm(enumerate(self.corpus), "Finding Similarity"):
            for j, label in enumerate(self.seedwords):
                self.relevance[i][j] = cosine_similarity(
                    self.doc_rep[i], self.label_rep[j]
                )
        
        # predict
        self.predictions = self.pred_to_label(self.relevance.argmax(axis=1))
        return self.predictions
    
    def get_embedding(self, word):
        self.check_fitted()
        
        if word in self.model.wv:
            return self.model.wv[word]
        
        stemmer = EnglishStemmer()
        if stemmer.stem(word) in self.model.wv:
            return self.model.wv[stemmer.stem(word)]
        
        if '_' in word:
            words = word.strip().split('_')
            return self.get_avg_embedding(words)
    
    def get_avg_embedding(self, words):
        lst = []
        for w in words:
            w_emb = self.get_embedding(w)
            if w_emb is not None:
                lst.append(w_emb)
        if not lst:
            return
        return np.vstack(lst).mean(axis=0)

    def get_max_sim_distribution(self):
        return pd.Series(self.relevance.max(axis=1)).plot(kind='hist')
    
    def confidence_split(self, threshold=0.1):
        """
        Get confident predictions, and unconfident corpus
        """
        unconfident = self.relevance.max(axis=1) < threshold
        conf_docs = [' '.join(doc) for doc, sim in zip(self.corpus, unconfident) if not sim]
        unconf_docs = [doc for doc, sim in zip(self.corpus, unconfident) if sim]
        conf_idxs = np.argwhere(~unconfident).flatten()
        conf_pred = self.predictions[conf_idxs]
        unconf_idxs = np.argwhere(unconfident).flatten()
        unconf_reps = self.doc_rep[unconfident]
        return {
            'confident': pd.DataFrame(
                {'sentence': conf_docs, 'predictions': conf_pred}, 
                index=conf_idxs
            ),
            'unconfident': (unconf_docs, unconf_idxs, unconf_reps)
        }
    
    def check_fitted(self):
        if not hasattr(self, "model"):
            raise NotFittedError('Please fit or load the model first')
    
    def save_model(self, out_path):
        assert '.model' in out_path
        self.model.save(out_path)
        write_config(self.config, path=out_path.replace('.model', '_config.json'))


class Word2Vec_Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        # loss = model.get_latest_training_loss()
        self.epoch += 1
        if self.epoch % 5 == 0:
            print(f'Epoch: {self.epoch}')


class Clustering_Model:
    def __init__(self, method, vectors, vector_idx):
        assert method.lower() in ['kmeans', 'gmm'], "Only supports KMeans and Gaussian Mixture"
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
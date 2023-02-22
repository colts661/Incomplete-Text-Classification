from collections import Counter
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

class TF_IDF_Model:
    """
    Basic TF-IDF model for most informative words in documents
    """

    def fit_transform(self, corpus: pd.DataFrame):
        """
        Learns TF-IDF values. Input corpus needs the `sentence` and `label`
        columns, and have each document on each row
        """
        corpus = (
            corpus
            .groupby('label', as_index=False)['sentence']
            .apply(lambda doc: ' '.join(doc))
        )
        self.model = TfidfVectorizer(stop_words='english')
        self.values_matrix = self.model.fit_transform(corpus['sentence'])
        self.vocabulary = self.model.get_feature_names_out()
        self.classes = corpus['label'].to_numpy()
    
    def get_top_dict(self, stop_words=None, k=10, strict=False, has_label=True):
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
        
        for class_idx, class_label in enumerate(self.classes):
            top_class_indices = np.flip(
                self.values_matrix[class_idx]
                .todense().A1
                .argsort()[-top_take:]
            )
            top = self.vocabulary[top_class_indices]
            if has_label:  # use label as first seed word
                top = np.concatenate((
                    [class_label.lower()],
                    top[top != class_label.lower()]
                ))
            top_dict[class_label] = top
            all_top.extend(top)
        
        # remove duplicates
        duplicates = {word for word, count in Counter(all_top).most_common() if count > 1}
        duplicates = duplicates.union(stop_words)

        for label, top_words in top_dict.items():
            if has_label:  # always keep class label
                top_dict[label] = ([top_words[0]] + [
                    word for word in top_words[1:] 
                    if word not in duplicates and word not in self.classes
                ][:k-1])
            else:
                top_dict[label] = ([
                    word for word in top_words 
                    if word not in duplicates
                ][:k])
        return top_dict


class LI_cTF_IDF_Model(TF_IDF_Model):
    """
    Implementation of the LI-cTF-IDF values
    """

    def fit_transform(self, corpus: pd.DataFrame):
        """
        Learn the LI-cTF-IDF values. Inputs corpus needs the `sentence` and 
        `label` columns, and have each document on each row
        """
        # produce raw frequency
        self.model = CountVectorizer(stop_words='english')
        freq_mtx = self.model.fit_transform(corpus['sentence'])
        self.vocabulary = self.model.get_feature_names_out()

        # produce |C|x|D| class assignment sparse matrix
        ohe = OneHotEncoder()
        assignment_mtx = ohe.fit_transform(corpus['label'].to_numpy().reshape(-1, 1)).T
        self.classes = ohe.categories_[0]
        doc_freq = 1 / assignment_mtx.sum(axis=1)

        # compute each component
        LI = (assignment_mtx @ (freq_mtx > 0)).multiply(doc_freq)
        c_TF = (assignment_mtx @ freq_mtx).multiply(doc_freq).tanh()
        IDF = np.log(freq_mtx.shape[0] / freq_mtx.sum(axis=0))
        self.values_matrix = LI.multiply(c_TF).multiply(IDF).power(1/3).tocsr()

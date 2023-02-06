"""
Data Loading and Corpus Generation
"""
import pickle
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer


class Data:
    """
    Data class
    """
    def __init__(
        self, data_dir, dataset, 
        stem=True, 
        random_seed=None, **remove_kwargs
    ):
        """
        Load and preprocess data.
        """
        # general info
        self.name = dataset
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # paths
        if dataset == 'testdata':
            self.raw_path = os.path.join(data_dir, dataset)
            self.processed_path = os.path.join(data_dir, dataset)
        else:
            self.raw_path = os.path.join(data_dir, 'raw', dataset)
            self.processed_path = os.path.join(data_dir, 'processed', dataset)
            if not os.path.exists(self.processed_path):
                os.mkdir(self.processed_path)
        
        # preprocessing choices
        self.stem = stem
        self.stemmer = EnglishStemmer().stem if stem else lambda word: word
        self.tokenizer = RegexpTokenizer('\w\w+')
        self.corpus_path = lambda s: os.path.join(
            self.processed_path, 
            f"{s}_corpus{'_stem' if stem else ''}.pkl"
        )
        self.labels_path = lambda s: os.path.join(
            self.processed_path, 
            f"{s}_labels.pkl"
        )

        # load corpus
        if os.path.exists(self.corpus_path('labeled')):
            self.labeled_corpus = pickle.load(open(self.corpus_path('labeled'), 'rb'))
            self.labeled_labels = pickle.load(open(self.labels_path('labeled'), 'rb'))
            self.unlabeled_corpus = pickle.load(open(self.corpus_path('unlabeled'), 'rb'))
            self.unlabeled_labels = pickle.load(open(self.labels_path('unlabeled'), 'rb'))
        else:
            self.preprocess(**remove_kwargs)

        # process labels
        self.labels = list(set(self.unlabeled_labels))
        self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])
    
    def preprocess(self, **remove_kwargs):
        """
        Preprocess corpus and store result into file
        ---
        Parameters:
        remove_kwargs: keyword arguments for remove_labels
        """
        print(f'Preprocessing Corpus: {self.name}')
        df = pickle.load(open(os.path.join(self.raw_path, "df.pkl"), "rb"))

        # tokenize, stem corpus
        df['sentence'] = (
            df['sentence']
            .str.lower()
            .apply(lambda s: [
                self.stemmer(w) 
                for w in self.tokenizer.tokenize(s)
            ])
        )
        
        self.labeled_corpus, self.labeled_labels, \
        self.unlabeled_corpus, self.unlabeled_labels = Data.remove_labels(df, **remove_kwargs)

        with open(self.corpus_path('labeled'), 'wb') as f:
            pickle.dump(self.labeled_corpus, f)

        if not os.path.exists(self.labels_path('labeled')):
            with open(self.labels_path('labeled'), 'wb') as f:
                pickle.dump(self.labeled_labels, f)
        
        with open(self.corpus_path('unlabeled'), 'wb') as f:
            pickle.dump(self.unlabeled_corpus, f)
        
        if not os.path.exists(self.labels_path('unlabeled')):
            with open(self.labels_path('unlabeled'), 'wb') as f:
                pickle.dump(self.unlabeled_labels, f)
        print('done\n')
    
    @staticmethod
    def remove_labels(df, bottom_p=0.5, full_k=5, keep_p=0.1, **kwargs):
        """
        Remove labels according to the incomplete setting.
        ---
        Parameters:
        df (pd.DataFrame): a table consisting `sentence`, `label` columns
        bottom_p (float(0, 1]): percentage of rarest labels sampled from to fully remove
        full_k (int): the number of labels to sample to fully remove
        keep_p (float(0, 0.5]): percentage of examples to keep for each remaining label
        ---
        Returns:
        list[str]: labeled documents
        np.ndarray[str]: labeled labels
        list[str]: unlabeled documents
        np.ndarray[str]: unlabeled labels (for evaluation only)
        """
        # count label frequency
        label_counts = df['label'].value_counts().sort_values()
        unpopular_labels = label_counts.iloc[:int(label_counts.shape[0] * bottom_p)]
        
        # sample fully removed label
        remove_labels = unpopular_labels.sample(n=full_k, replace=True, weights=1/unpopular_labels)
        removed_full = df[~df['label'].isin(remove_labels.index)]
        
        # sample partially removed label
        removed_by_frac = removed_full.groupby('label').sample(frac=keep_p, replace=False)
        
        # create two sets of documents
        labeled = df.loc[df.index.isin(removed_by_frac.index)].reset_index()
        removed = df.loc[~df.index.isin(removed_by_frac.index)].reset_index()
        
        return (
            labeled['sentence'].tolist(), labeled['label'].to_numpy(),
            removed['sentence'].tolist(), removed['label'].to_numpy()
        )

    def __repr__(self):
        return f"{self.name} Dataset{' (Stemmed)' if self.stem else ''}"

if __name__ == '__main__':
    # run main in `src` folder
    dataset = sys.argv[1]
    d = Data(
        data_dir='../data' if dataset != 'testdata' else '../test',
        dataset=dataset,
        stem=False,
        random_seed=42
    )

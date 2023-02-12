"""
Data Loading and Corpus Generation
"""
import pickle
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)


class Data:
    """
    Data class
    """

    def __init__(self, data_dir, dataset):
        # general info
        self.name = dataset
        
        # paths
        self.raw_path = os.path.join(data_dir, 'raw', dataset)
        self.processed_path = os.path.join(data_dir, 'processed', dataset)
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        self.corpus_path = lambda s: os.path.join(self.processed_path, f"{s}_corpus.pkl")
        self.labels_path = lambda s: os.path.join(self.processed_path, f"{s}_labels.pkl")

    def process_corpus(
        self, remove_punctuation=True, remove_stopwords=True,
        stem=False, lemmatize=False, 
        word_piece=False, phrase_mine=False  # more advanced, not implemented yet
    ):
        data_config = {
            'remove_punctuation': remove_punctuation,
            'remove_stopwords': remove_stopwords,
            'stem': stem,
            'lemmatize': lemmatize,
            'word_piece': word_piece,
            'phrase_mine': phrase_mine
        }
        load_config = self.read_current_config('data')

        if load_config is None or load_config != data_config:
            print(f'Preprocessing Corpus: {self.name}')
            self.df = pickle.load(open(os.path.join(self.raw_path, "df.pkl"), "rb"))
            if remove_punctuation:
                self.df['sentence'] = self.df['sentence'].str.replace('[^\w\s]', '', regex=True)

            # tokenize
            self.df['sentence'] = self.df['sentence'].apply(lambda s: word_tokenize(s.lower()))
            
            if remove_stopwords:
                english_stopwords = stopwords.words('english')
                self.df['sentence'] = self.df['sentence'].apply(
                    lambda s: list(filter(lambda w: w not in english_stopwords, s))
                )

            if stem:
                stemmer = EnglishStemmer()
                self.df['sentence'] = (
                    self.df['sentence']
                    .apply(lambda s: [stemmer.stem(w) for w in s])
                )

            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                self.df['sentence'] = (
                    self.df['sentence']
                    .apply(lambda s: [lemmatizer.lemmatize(w) for w in s])
                )
            
            if word_piece:
                print("Not Implemented Yet")
                pass

            if phrase_mine:
                print("Not Implemented Yet")
                pass

            self.write_config('data', data_config)
            self.write_pickle(os.path.join(self.processed_path, 'df.pkl'), self.df)
            print('Done, please run process_label with rerun=True\n')
        else:
            print(f'Loading Corpus: {self.name}')
            self.df = self.read_pickle(os.path.join(self.processed_path, 'df.pkl'))
            self.labeled_corpus = self.read_pickle(self.corpus_path('labeled'))
            self.labeled_labels = self.read_pickle(self.labels_path('labeled'))
            self.unlabeled_corpus = self.read_pickle(self.corpus_path('unlabeled'))
            self.unlabeled_labels = self.read_pickle(self.labels_path('unlabeled'))
            
            # process labels
            self.labels = list(set(self.unlabeled_labels))
            self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])
            print('Done\n')
        
    def process_labels(self, bottom_p=0.5, full_k=5, keep_p=0.1, random_seed=None, rerun=False):
        """
        Remove labels according to the incomplete setting
        ---
        Parameters:
        bottom_p (float(0, 1]): percentage of rarest labels sampled from to fully remove
        full_k (int): the number of labels to sample to fully remove
        keep_p (float(0, 0.5]): percentage of examples to keep for each remaining label
        random_seed (int): the deterministic random seed for pseudo random numbers
        rerun (bool): indicating whether to force rerun
        """
        if not hasattr(self, 'df'):
            raise ValueError('Corpus not loaded, please run process_corpus')
        
        if random_seed is None:
            random_seed = 42

        # check if preload
        label_config = {
            'bottom_p': bottom_p,
            'full_k': full_k,
            'keep_p': keep_p,
            'random_seed': random_seed
        }
        load_config = self.read_current_config('label')

        if rerun or load_config is None or load_config != label_config:
            # count label frequency
            label_counts = self.df['label'].value_counts().sort_values()
            unpopular_labels = label_counts.iloc[:int(label_counts.shape[0] * bottom_p)]
            
            # sample fully removed label
            remove_labels = unpopular_labels.sample(
                n=full_k, replace=True, weights=1/unpopular_labels, 
                random_state=random_seed
            )
            removed_full = self.df[~self.df['label'].isin(remove_labels.index)]
            
            # sample partially removed label
            removed_by_frac = removed_full.groupby('label').sample(
                frac=keep_p, replace=False, random_state=random_seed
            )
            
            # create two sets of documents
            labeled = self.df.loc[self.df.index.isin(removed_by_frac.index)].reset_index()
            removed = self.df.loc[~self.df.index.isin(removed_by_frac.index)].reset_index()
            
            # load attributes
            self.labeled_corpus = labeled['sentence'].tolist()
            self.labeled_labels = labeled['label'].to_numpy()
            self.unlabeled_corpus = removed['sentence'].tolist()
            self.unlabeled_labels = removed['label'].to_numpy()

            # write to file
            self.write_config('label', label_config)
            self.write_pickle(self.corpus_path('labeled'), self.labeled_corpus)
            self.write_pickle(self.labels_path('labeled'), self.labeled_labels)
            self.write_pickle(self.corpus_path('unlabeled'), self.unlabeled_corpus)
            self.write_pickle(self.labels_path('unlabeled'), self.unlabeled_labels)

            # process labels
            self.labels = list(set(self.unlabeled_labels))
            self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])
        else:
            print('Changes not detected, no rerun needed')

    def read_current_config(self, mode):
        """
        Reads the saved config file into a dictionary, or return None
        if it does not exist
        """
        assert mode in ['label', 'data']
        try:
            with open(os.path.join(self.processed_path, f"{mode}_config.json"), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return
    
    def write_config(self, mode, config: dict):
        assert mode in ['label', 'data']
        with open(os.path.join(self.processed_path, f"{mode}_config.json"), 'w') as f:
            json.dump(config, f)

    def read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def write_pickle(self, path, content):
        with open(path, 'wb') as f:
            pickle.dump(content, f)
    
    def __repr__(self):
        data_config = dict(tup for tup in self.read_current_config('data').items() if tup[1])
        return (
            f"## {self.name} Dataset" +
            f"\nPreprocessing:\n{data_config}" +
            f"\nLabel Removal:\n{self.read_current_config('label')}"
        )
    
    def show_statistics(self):
        return pd.DataFrame(
            [
                [
                    len(self.labeled_labels),
                    sum(len(doc) for doc in self.labeled_corpus) / len(self.labeled_corpus),
                    len(set(self.labeled_labels))
                ],
                [
                    len(self.unlabeled_labels),
                    sum(len(doc) for doc in self.unlabeled_corpus) / len(self.unlabeled_corpus),
                    len(set(self.unlabeled_labels))
                ]
            ],
            columns=['Size', 'Average Length', 'Number of Classes'],
            index=['Labeled', 'Unlabeled']
        )

if __name__ == '__main__':
    # run main in `src` folder
    dataset = sys.argv[1]
    d = Data(
        data_dir='../data' if dataset != 'testdata' else '../test',
        dataset=dataset
    )
    d.process_corpus()
    d.process_labels()

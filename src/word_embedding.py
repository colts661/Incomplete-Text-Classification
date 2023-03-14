"""
A collection of word embedding models
"""

# basic packages
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project dependencies
import data
import util

# sklearn
from sklearn.exceptions import NotFittedError

# Word2Vec packages
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import gensim.downloader as gensim_api

# BERT packages
import torch

# misc
import os
import string
from tqdm import tqdm
from nltk.stem.snowball import EnglishStemmer


class Word_Embedding_Model:
    """
    General word embedding model. Needs to be inherited
    """

    def __init__(self, data: data.Data):
        self.data = data
        self.corpus = self.data.full_corpus           
    
    def fit(self, **config):
        raise NotImplementedError
    
    def load_model(self, path_or_model):
        raise ValueError('This model does not support loading')

    def get_document_embeddings(self):
        raise NotImplementedError
    
    def get_class_embeddings(self, seed_words):
        raise NotImplementedError
    
    def check_fitted(self):
        if not hasattr(self, "model"):
            raise NotFittedError('Please fit or load the model first')
    
    def save_model(self, out_path, suffix='.model'):
        assert suffix in out_path
        if isinstance(self.model, dict):
            util.write_pickle(out_path, self.model)
        else:
            self.model.save(out_path)
        util.write_config(self.config, path=out_path.replace(suffix, '_config.json'))


class Word2Vec_Model(Word_Embedding_Model):
    """
    Word2Vec Embedding based on window context of a word
    """
    def fit(self, **config):
        """
        Train the Word2Vec model using config in model.
        """
        self.config = config
        self.model = Word2Vec(self.corpus, **config, callbacks=[Word2Vec_Callback()])
        self.dimension = config['vector_size'] if 'vector_size' in config else 100

    def load_model(self, path_or_model):
        if os.path.exists(path_or_model):
            assert '.model' in path_or_model
            self.model = Word2Vec.load(path_or_model)
            self.config = util.load_config(path_or_model.replace('.model', '_config.json'))
        elif isinstance(path_or_model, str):
            try:
                self.model = gensim_api.load(path_or_model)
            except ValueError:
                pass
        else:
            raise FileNotFoundError('Model or path to model not found')
    
    def get_word_embedding(self, word):
        self.check_fitted()
        
        if word in self.model.wv:
            return self.model.wv[word]
        
        stemmer = EnglishStemmer()
        if stemmer.stem(word) in self.model.wv:
            return self.model.wv[stemmer.stem(word)]
        
        if '_' in word:
            words = word.strip().split('_')
            return self.get_document_embedding(words)

    def get_document_embedding(self, document):
        lst = []
        for w in document:
            w_emb = self.get_word_embedding(w)
            if w_emb is not None:
                lst.append(w_emb)
        if not lst:
            return
        return np.vstack(lst).mean(axis=0)
    
    def get_document_embeddings(self):
        doc_rep = np.empty((len(self.corpus), self.dimension))
        for i, doc in tqdm(
            enumerate(self.corpus), 
            "Finding Document Representations"
        ):
            embed = self.get_document_embedding(doc)
            if embed is not None:
                doc_rep[i] = embed
        return doc_rep
    
    def get_class_embeddings(self, seed_words: dict) -> dict:
        print("Finding Class Representations\n")
        class_rep = dict()
        for cls, seeds in tqdm(seed_words.items()):
            embed = self.get_document_embedding(seeds)
            if embed is not None:
                class_rep[cls] = embed
        return class_rep


class Word2Vec_Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % 5 == 0:
            print(f'Epoch: {self.epoch}')


class BERT_Embedding(Word_Embedding_Model):
    """
    Contextualized BERT Embedding. Code adapted from https://github.com/ZihanWangKi/XClass
    """
    def fit(self, tokenizer, model):
        """
        Find contextualized word embedding. 
        """
        self.dimension = model.config.hidden_size
        # get counts to eliminate words
        tokenization_info = []
        counts = Counter()
        for text in tqdm(self.corpus, "Obtaining Counts"):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = self.prepare_sentence(tokenizer, text)
            counts.update(word.translate(str.maketrans('','', string.punctuation)) for word in tokenized_text)
        del counts['']

        # get all occurrences of contextualized words
        updated_counts = {k: c for k, c in counts.items() if c >= 5}
        word_rep = {}
        word_count = {}
        for text in tqdm(self.corpus, "Collecting Contextualized Embeddings"):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = self.prepare_sentence(tokenizer, text)
            tokenization_info.append((tokenized_text, tokenized_to_id_indicies, tokenids_chunks))
            contextualized_word_representations = self.handle_sentence(model, 12, tokenized_text,
                                            tokenized_to_id_indicies, tokenids_chunks)
            for i in range(len(tokenized_text)):
                word = tokenized_text[i]
                if word in updated_counts.keys():
                    if word not in word_rep:
                        word_rep[word] = 0
                        word_count[word] = 0
                    word_rep[word] += contextualized_word_representations[i]
                    word_count[word] += 1

        # average occurrences
        self.model = {}
        for k,v in tqdm(word_rep.items(), "Computing Word Embeddings"):
            self.model[k] = word_rep[k]/word_count[k]
        
        # save intermediate files
        vocab_words = list(self.model.keys())
        static_word_representations = list(self.model.values())
        vocab_occurrence = list(word_count.values())

        util.write_pickle(
            os.path.join(self.data.processed_path, f"tokenization_lm-uncased-12.pk"),
            {"tokenization_info": tokenization_info}
        )

        util.write_pickle(
            os.path.join(self.data.processed_path, f"static_repr_lm-uncased-12.pk"),
            {
                "static_word_representations": static_word_representations,
                "vocab_words": vocab_words,
                "word_to_index": {v: k for k, v in enumerate(vocab_words)},
                "vocab_occurrence": vocab_occurrence,
            }
        )
        
    def get_document_embeddings(self):
        try: # get fitted word embeddings
            self.check_fitted()
        except NotFittedError:
            vocab = util.read_pickle(os.path.join(data.processed_path, f"static_repr_lm-uncased-12.pk"))
            static_word_representations = vocab["static_word_representations"]
            vocab_words = vocab["vocab_words"]
            self.model = {w: rep for w, rep in zip(vocab_words, static_word_representations)}
   
        # Evaluate the model in batch mode
        embeddings = []
        with torch.no_grad():
            for sent in tqdm(self.corpus, "Finding Document Embeddings"):
                one_sent = [self.model[w] if w in self.model else np.zeros(768) for w in sent.split(' ')]
                embeddings.append(torch.tensor(np.vstack(one_sent).mean(axis=0)).float().to(util.DEVICE))

        # Concatenate the embeddings   
        return torch.vstack(embeddings)

    def get_class_embeddings(self, seed_words: dict) -> dict:
        class_embeddings = {}
        for c, seeds in seed_words.items():
            print(c, seeds)
            one_class_embed = []
            for seed_word in seeds:
                if seed_word in self.model:
                    one_class_embed.append(self.model[seed_word])
            print(one_class_embed)
            class_embeddings[c] = np.vstack(one_class_embed).mean(axis=0)
        return class_embeddings
        

    def prepare_sentence(self, tokenizer, text):
        # setting for BERT
        model_max_tokens = 512
        has_sos_eos = True
        ######################
        max_tokens = model_max_tokens
        if has_sos_eos:
            max_tokens -= 2
        sliding_window_size = max_tokens // 2

        if not hasattr(self.prepare_sentence, "sos_id"):
            self.sos_id, self.eos_id = tokenizer.encode("", add_special_tokens=True)

        tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
        tokenized_to_id_indicies = []

        tokenids_chunks = []
        tokenids_chunk = []

        for index, token in enumerate(tokenized_text + [None]):
            if token is not None:
                tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
                tokenids_chunks.append([self.sos_id] + tokenids_chunk + [self.eos_id])
                if sliding_window_size > 0:
                    tokenids_chunk = tokenids_chunk[-sliding_window_size:]
                else:
                    tokenids_chunk = []
            if token is not None:
                tokenized_to_id_indicies.append((len(tokenids_chunks),
                                                len(tokenids_chunk),
                                                len(tokenids_chunk) + len(tokens)))
                tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

        return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


    def sentence_encode(self, tokens_id, model, layer):
        input_ids = torch.tensor([tokens_id], device=model.device)

        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]

        layer_embedding = util.tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
        return layer_embedding


    def sentence_to_wordtoken_embeddings(self, layer_embeddings, tokenized_text, tokenized_to_id_indicies):
        word_embeddings = []
        for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
            word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
        assert len(word_embeddings) == len(tokenized_text)
        return np.array(word_embeddings)


    def handle_sentence(self, model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
        layer_embeddings = [
            self.sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
        ]
        word_embeddings = self.sentence_to_wordtoken_embeddings(
            layer_embeddings, tokenized_text, tokenized_to_id_indicies
        )
        return word_embeddings


    def collect_vocab(self, token_list, representation, vocab):
        assert len(token_list) == len(representation)
        for token, repre in zip(token_list, representation):
            if token not in vocab:
                vocab[token] = []
            vocab[token].append(repre)


    def estimate_static(self, vocab, vocab_min_occurrence):
        static_word_representation = []
        vocab_words = []
        vocab_occurrence = []
        for word, repr_list in tqdm(vocab.items(), total=len(vocab)):
            if len(repr_list) < vocab_min_occurrence:
                continue
            vocab_words.append(word)
            vocab_occurrence.append(len(repr_list))
            static_word_representation.append(np.average(repr_list, axis=0))
        static_word_representation = np.array(static_word_representation)
        print(f"Saved {len(static_word_representation)}/{len(vocab)} words.")
        return static_word_representation, vocab_words, vocab_occurrence

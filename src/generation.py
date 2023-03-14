"""
Seed Word/Label Generation Module
"""

from collections import Counter
import numpy as np
import pandas as pd
import re
import openai
import backoff
import string

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
import data

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


def get_seed_words(data: data.Data, k=10):
    df_labeled = pd.DataFrame({'sentence': [
        ' '.join(doc) for doc in data.labeled_corpus
    ], 'label': data.labeled_labels})

    tfidf = TF_IDF_Model()
    tfidf.fit_transform(df_labeled)
    return tfidf.get_top_dict(k=k, has_label=False)


def get_li_tfidf_class_label(data: data.Data, unconfident_idx, unconfident_docs, cluster_results):
    df_new_classes = pd.DataFrame({'sentence': [
        ' '.join(doc) for doc in unconfident_docs.values()
    ], 'label': cluster_results}, index=unconfident_idx)

    li_ctf_idf_clusters = LI_cTF_IDF_Model()
    li_ctf_idf_clusters.fit_transform(df_new_classes)
    cluster_labels = li_ctf_idf_clusters.get_top_dict(k=8, has_label=False)
    predict_cluster_word_tfidf = {cluster_id: top[0] for cluster_id, top in cluster_labels.items()}
    return predict_cluster_word_tfidf


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_gpt_completion(prompt_type, content, seen_labels):
    print("Please store your `openai` API key in the variable `openai.api_key`")
    doc_prompt = "Generate a general topic for the following document. No header needed."
    full_prompt_1 = "Use one generic topic word to describe the following list of topics."
    full_prmopt_2 = f"Example labels: {', '.join(seen_labels)}."
    full_prompt_3 = "The label should be with similar granuality of, but different from the examples."
    full_prmopt_4 = "Your answer should limit to one single word."
    
    if prompt_type == 'doc':
        prompt = doc_prompt
    elif prompt_type == 'label':
        prompt = full_prompt_1 + full_prmopt_2 + full_prompt_3 + full_prmopt_4
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a helpful text summarizer and topic generator."},
          {"role": "user", "content": prompt + content},
      ]
    )
    
    s = re.sub(pattern="\"|'", repl="", string=response.choices[0]['message']['content'])
    return s.translate(str.maketrans("", "", string.punctuation))


def get_gpt_label(data: data.Data, label_samples, unconfident_docs):
    predict_cluster_label = {}
    for cluster, doc_indices in label_samples.items():
        all_topics = []
        for idx, doc_idx in enumerate(doc_indices):
            document = ' '.join(unconfident_docs[doc_idx])
            all_topics.append(f'{idx+1}. ' + get_gpt_completion('doc', document, data.existing_labels))
        
        all_topics = '\n'.join(all_topics)
        predict_cluster_label[cluster] = get_gpt_completion('label', all_topics, data.existing_labels)
    return predict_cluster_label


def get_full_prediction(df_new_classes, predict_cluster_label, confident_predictions):
    new_predictions = df_new_classes.assign(
        predictions=df_new_classes['label'].apply(lambda cid: predict_cluster_label[cid])
    )
    full_pred = pd.concat([
        new_predictions[['sentence', 'predictions']], 
        confident_predictions[['sentence', 'predictions']]
    ]).sort_index()

    return full_pred

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
import openai
from data import Data
import evaluation, generation, similarity, unsupervised, util, word_embedding
from transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings('ignore')


def run_model(dataset, model_type):
    if not os.path.exists('artifacts'):
        os.mkdir('artifacts')
    now = datetime.now().strftime('%Y-%m-%d')
    
    # load data
    d = Data('../data', dataset)
    d.process_corpus()
    d.process_labels()

    if model_type == 'final':
        # load pretrained BERT
        bert_model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        model.eval()
        model.to(util.DEVICE)

    # get seed words
    seed_words = generation.get_seed_words(
        d,
        k=(1 if dataset == 'testdata' else 10),
        strict=(dataset == 'testdata')
    )

    if model_type == 'final':
        # find contexualized BERT embeddings
        bert_embedding = word_embedding.BERT_Embedding(d)
        bert_embedding.fit(tokenizer, model)
    elif model_type == 'baseline':
        w2v = Word2Vec_Model(corpus=self.data.unlabeled_corpus, seedwords=seed_words)
        
        w2v_full_config = {
            "alpha": 0.01, 
            "window": 10, 
            "min_count": 10, 
            "sample": 0.001, 
            "seed": 42, 
            "sg": 0,
            "hs": 1
        }
        if w2v_config is None:
            w2v_full_config["vector_size"] = 128
            w2v_full_config["epoch"] = 125
        else:
            w2v_full_config.update(w2v_config)
        
        if w2v_model is not None and os.path.exists(w2v_model):
            w2v.load_model(w2v_model)
        else:
            w2v.fit(**w2v_full_config)
            w2v.save_model(f"artifacts/{self.data.name}_baseline_{now}.model")
        
        w2v_pred = w2v.predict()

    # get doc/class representations
    doc_rep = bert_embedding.get_document_embeddings()
    class_rep = bert_embedding.get_class_embeddings(seed_words)

    if model_type == 'final':
        # run PCA
        rep_pca = unsupervised.Dimensionality_Reduction('pca')
        low_dim_doc_rep = rep_pca.fit_transform(util.tensor_to_numpy(doc_rep), dimension=128)
        low_dim_class_rep = rep_pca.transform(class_rep)
    else:
        low_dim_doc_rep, low_dim_class_rep = doc_rep, class_rep

    # run similarity
    max_sim, argmax_sim = similarity.get_cosine_similarity_batched(
        d, low_dim_doc_rep[len(d.labeled_labels):], low_dim_class_rep
    )
    sim_fig = similarity.plot_max_similarity(max_sim)
    sim_fig.savefig(f'artifacts/{dataset}_{model_type}_sim_distribution_{now}.png')

    # split by confidence
    threshold = input("Enter a threshold for unconfidence: ")
    split_result = similarity.confidence_split(
        max_sim, argmax_sim, d, low_dim_doc_rep, threshold=float(threshold)
    )
    unconfident_docs, unconfident_idx, unconfident_rep = split_result['unconfident']
    confident_predictions = split_result['confident']
    tsne_fig = similarity.display_vectors(d, unconfident_rep, unconfident_idx, pca_dim=50, tsne_perp=30)
    tsne_fig.savefig(f'artifacts/{dataset}_{model_type}_tsne_distribution_{now}.png')

    # run cluster
    print("Running Clustering")
    gmm = unsupervised.Clustering_Model('gmm', unconfident_rep, unconfident_idx)
    gmm_results = gmm.fit_transform(n_classes=5)

    # run label generation
    print("For time and cost purpose, run LI-TF-IDF Label generation")
    li_tfidf_labels = generation.get_li_tfidf_class_label(d, unconfident_idx, unconfident_docs, gmm_results)
    
    # evaluation
    df_new_classes = generation.get_df_new_classes(d, unconfident_docs, unconfident_idx, gmm_results, li_tfidf_labels)
    full_pred = generation.get_full_prediction(df_new_classes, li_tfidf_labels, confident_predictions)
    e = evaluation.Evaluation(d, full_pred, list(li_tfidf_labels.values()))
    print("Evaluations:")
    print(f"New Label Binary: {e.evaluate_new_label_metrics()}")
    print()

    print(f"Existing Label Performances: {e.evaluate_existing()}")
    print()
    
    print(f"Word Cloud: Left Original; Right Generated")
    wc_fig = e.plot_word_cloud(df_new_classes)
    wc_fig.savefig(f'artifacts/{dataset}_{model_type}_word_cloud_{now}.png')


def run_final_model(dataset):
    run_model(dataset, model_type='final')


def run_baseline_model(dataset):
    run_model(dataset, model_type='baseline')

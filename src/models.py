import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    print(f"Running {model_type} model on {dataset}")
    print("Enter user input when prompted. Figures are stored in `artifacts/`")
    if dataset != "testdata":
        print("This might take some time to run.")
    
    # load data
    if dataset == "testdata":
        d = Data('test', 'testdata')
        d.process_corpus(remove_stopwords=False)
        d.process_labels(bottom_p=0.5, full_k=1, keep_p=0.6)
    else:
        d = Data('data', dataset)
        d.process_corpus()
        d.process_labels()
    print(d.show_statistics())

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
        k=(3 if dataset == 'testdata' else 10),
        strict=(dataset == 'testdata')
    )

    if model_type == 'final':
        # find contexualized BERT embeddings
        model_pipeline = word_embedding.BERT_Embedding(d)
        model_pipeline.fit(tokenizer, model)
    elif model_type == 'baseline':
        model_pipeline = word_embedding.Word2Vec_Model(d)
        
        if dataset == 'testdata':
            w2v_config = {
                'vector_size': 16,
                'epochs': 2,
                'window': 3,
                'min_count': 1
            }
            model_pipeline.load_model('glove-twitter-25')
        else:
            w2v_config = {
                "vector_size": 64, 
                "alpha": 0.01, 
                "window": 10, 
                "min_count": 10, 
                "sample": 0.001, 
                "seed": 42, 
                "sg": 0, 
                "hs": 1, 
                "epochs": 2
            }
            model_pipeline.fit(**w2v_config)
            model_pipeline.save_model(f"artifacts/{dataset}_baseline_{now}.model")

    # get doc/class representations
    doc_rep = model_pipeline.get_document_embeddings()
    class_rep = model_pipeline.get_class_embeddings(seed_words)

    if model_type == 'final':
        # run PCA
        print("Running Dimensionality Reduction")
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
    similarity.max_similarity_histogram(max_sim)

    # split by confidence
    threshold = input("Enter a threshold for unconfidence: ")
    split_result = similarity.confidence_split(
        max_sim, argmax_sim, d, low_dim_doc_rep, threshold=float(threshold)
    )
    unconfident_docs, unconfident_idx, unconfident_rep = split_result['unconfident']
    confident_predictions = split_result['confident']

    if dataset != "testdata":
        tsne_fig = similarity.display_vectors(d, unconfident_rep, unconfident_idx, pca_dim=50, tsne_perp=30)
        tsne_fig.savefig(f'artifacts/{dataset}_{model_type}_tsne_distribution_{now}.png')

    # run cluster
    print("Running Clustering")
    if dataset != "testdata":
        gmm = unsupervised.Clustering_Model('gmm', unconfident_rep, unconfident_idx)
        gmm_results = gmm.fit_transform(n_classes=5 if dataset != "testdata" else 1)

    # run label generation
    print("For time and cost purposes, run LI-TF-IDF Label generation")
    li_tfidf_labels = generation.get_li_tfidf_class_label(d, unconfident_idx, unconfident_docs, gmm_results)
    print()
    
    # evaluation
    df_new_classes = generation.get_df_new_classes(d, unconfident_docs, unconfident_idx, gmm_results, li_tfidf_labels)
    full_pred = generation.get_full_prediction(df_new_classes, li_tfidf_labels, confident_predictions)
    e = evaluation.Evaluation(d, full_pred, list(li_tfidf_labels.values()))
    print("Evaluations:")
    print(f"New Label Binary: {e.evaluate_new_label_metrics()}")
    print(f"Existing Label Performances: {e.evaluate_existing()}")

    if dataset != "testdata":
        wc_fig = e.plot_word_cloud(df_new_classes)
        wc_fig.savefig(f'artifacts/{dataset}_{model_type}_word_cloud_{now}.png')


def run_final_model(dataset):
    run_model(dataset, model_type='final')


def run_baseline_model(dataset):
    run_model(dataset, model_type='baseline')

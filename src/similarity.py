"""
Similarity Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

import util, data
from unsupervised import Dimensionality_Reduction
from tqdm import tqdm

class RepDataset(Dataset):
    """
    Representation Dataset, supporting large dimension vectors
    """
    def __init__(self, reps):
        self.reps = reps

    def __len__(self):
        return self.reps.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx]


class CosineSimilarityModel(torch.nn.Module):
    """
    Batching cosine similarity
    """
    def __init__(self, cls_mtx):
        super(CosineSimilarityModel, self).__init__()
        if not isinstance(cls_mtx, torch.Tensor):
            cls_mtx = torch.tensor(cls_mtx)
        self.cls_mtx_norm = torch.nn.functional.normalize(cls_mtx.to(util.DEVICE).float(), dim=1)

    def forward(self, doc_mtx, self_sim=False, top_k=8):
        doc_mtx_norm = torch.nn.functional.normalize(doc_mtx, dim=1)
        similarities = torch.mm(doc_mtx_norm, self.cls_mtx_norm.t())
        
        if self_sim:
            return similarities.topk(k=top_k, dim=1)
        else:
            return similarities.max(dim=1)


def get_cosine_similarity_batched(data: data.Data, doc_mtx, cls_mtx, self_sim=False, batch_size=32, top_k=8):
    if isinstance(doc_mtx, dict):
        doc_mtx = np.vstack([doc_mtx[c] for c in data.existing_labels])
    rep_dataset = RepDataset(doc_mtx)
    dataloader = DataLoader(rep_dataset, batch_size=batch_size)
    
    model = CosineSimilarityModel(cls_mtx)
    max_sim, argmax_sim = [], []
    
    with torch.no_grad():
        for X in tqdm(dataloader, "Finding Cosine Similarity"):
            batch_max, batch_argmax = model(X.to(util.DEVICE).float(), self_sim, top_k)
            max_sim.append(batch_max)
            argmax_sim.append(batch_argmax)
    
    return util.tensor_to_numpy(torch.cat(max_sim)), util.tensor_to_numpy(torch.cat(argmax_sim))


def plot_max_similarity(max_similarity):
    """
    Plot the histogram of maximum similarity
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    pd.Series(max_similarity).plot(kind='hist', ax=ax)
    return fig


def confidence_split(max_sim, argmax_sim, data: data.Data, rep, threshold=0.1):
    """
    Get confident predictions, and unconfident corpus
    """
    unconfident = max_sim < threshold
    conf_docs = [' '.join(doc) for doc, sim in zip(data.unlabeled_corpus, unconfident) if not sim]
    unconf_docs = {idx: doc for idx, (doc, sim) in enumerate(zip(data.unlabeled_corpus, unconfident)) if sim}
    conf_idxs = np.argwhere(~unconfident).flatten()
    conf_pred = argmax_sim[conf_idxs]
    unconf_idxs = np.argwhere(unconfident).flatten()
    unconf_reps = rep[len(data.labeled_labels)+unconf_idxs]
    return {
        'confident': pd.DataFrame(
            {'sentence': conf_docs, 'predictions': conf_pred}, 
            index=conf_idxs
        ),
        'unconfident': (unconf_docs, unconf_idxs, unconf_reps)
    }


def display_vectors(data: data.Data, reps, idx, pca_dim=50, tsne_perp=30):
    """
    Displays the vector representations in 2D.
    Alert: This takes about 2h to run for ~100,000 documents on dimension 128.
    """
    # reduce dimension with PCA
    pca = Dimensionality_Reduction('pca')
    pca_rep = pca.fit_transform(reps, dimension=pca_dim)

    # t-SNE visualization transformation
    tsne = Dimensionality_Reduction('tsne')
    tsne_rep = tsne.fit_transform(reps, dimension=2, perplexity=tsne_perp)

    # preparation
    unconfident_truth = data.unlabeled_labels[idx]
    unconfident_truth_trun = [l if l in data.unseen_labels else 'Others' for l in unconfident_truth]
    plot_trun = pd.DataFrame({
        'x': tsne_rep[:, 0],
        'y': tsne_rep[:, 1],
        'label': unconfident_truth_trun
    })
    all_categories = np.unique(unconfident_truth_trun)

    # define colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    trun_colors = {
        lab: colors[i % len(colors)] if i != len(all_categories) - 1
        else '#b5b5b5'
        for i, lab in enumerate(all_categories)
    }
    
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    groups = plot_trun.groupby('label')
    for name, group in groups:
        ax.scatter(group.x, group.y, s=2, label=name, c=trun_colors[name], alpha=0.5)
    ax.legend()
    ax.set_title('Unconfident t-SNE Representation, Perplexity= 30')

    return fig

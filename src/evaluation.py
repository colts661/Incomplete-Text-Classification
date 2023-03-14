"""
Module for evaluating
"""

import data
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import f1_score, precision_recall_fscore_support

class Evaluation:
    """
    Evaluate the model based on 3 criteria
    """

    def __init__(self, data: data.Data, predictions, new_labels):
        assert len(data.unlabeled_labels) == len(predictions)
        self.data = data
        self.full_df = predictions.assign(truth=data.unlabeled_labels)
        self.new_labels = new_labels

        # existing labels: take existing labels from ground truth
        self.existing_idx, self.existing_pred, self.existing_label = zip(*[
            (idx, pred, truth) 
            for idx, (pred, truth) in enumerate(zip(self.full_df['predictions'], data.unlabeled_labels))
            if truth in self.existing_labels
        ])

        # removed label: to plot
        self.removed_idx, self.removed_doc, self.removed_label = zip(*[
            (idx, ' '.join(doc), truth) 
            for idx, (doc, truth) in enumerate(zip(data.unlabeled_corpus, data.unlabeled_labels))
            if truth not in data.existing_labels
        ])

    def evaluate_existing(self):
        """
        Evaluate the existing labels only
        """
        micro = f1_score(
            y_true=self.existing_label, y_pred=self.existing_pred, 
            labels=list(self.existing_labels), average='micro'
        )
        macro = f1_score(
            y_true=self.existing_label, y_pred=self.existing_pred, 
            labels=list(self.existing_labels), average='macro'
        )
        return {'micro_f1': round(micro, 3), 'macro_f1': round(macro, 3)}

    def evaluate_new_label_metrics(self):
        """
        Evaluate the binary case: whether label is new. Only returns
        the precision, recall for the new labels.
        """
        binary = self.full_df.isin(self.existing_labels).astype(int)
        prec, recall, _, _ = precision_recall_fscore_support(
            y_true=binary['truth'], y_pred=binary['predictions'],
            average=None
        )
        return {'precision': round(prec[0], 3), 'recall': round(recall[0], 3)}

    def compare_label_generation(self):
        """
        Directly compare removed and generated labels
        """
        print(f"Removed Labels: {list(self.data.unseen_labels)}\nGenerated Labels: {list(self.new_labels)}")

    def plot_word_cloud(self, df_new_classes):
        """
        Plots the word cloud comparing removed and generated label clouds
        """
        # process corpus
        combined_text_by_clusters = (
            df_new_classes.groupby('label')['sentence']
            .apply(lambda lst: ' '.join(lst)).to_frame()
        ).reset_index()

        removed_labels_df = pd.DataFrame({
            'label': self.removed_label,
            'sentence': self.removed_doc
        })
        combined_text_by_removed = (
            removed_labels_df.groupby('label')['sentence']
            .apply(lambda lst: ' '.join(lst)).to_frame()
        ).reset_index()

        # initiate word cloud and plot
        wc = WordCloud(
            background_color = "white",
            colormap = "Dark2", 
            max_font_size = 150,
            random_state = 42
        )

        fig, ax = plt.subplots(
            max(combined_text_by_removed.shape[0], combined_text_by_clusters.shape[0]), 2, 
            figsize=(24, 36)
        )

        for idx, (label, sentence) in combined_text_by_removed.iterrows():
            wc.generate(sentence)
            ax[idx][0].imshow(wc, interpolation='bilinear')
            ax[idx][0].axis('off')
            ax[idx][0].set_title(label, fontsize=45)

        for idx, (label, sentence) in combined_text_by_clusters.iterrows():
            wc.generate(sentence)
            ax[idx][1].imshow(wc, interpolation='bilinear')
            ax[idx][1].axis('off')
            ax[idx][1].set_title(label, fontsize=45)

        return fig

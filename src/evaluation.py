from util import *
from data import *
import gensim.downloader as gensim_api
from sklearn.metrics import f1_score, precision_recall_fscore_support

class Evaluation:
    """
    Evaluate the model based on similarity mapping and macro-F1 score.
    Assuming all suggested classes can be mapped to removed labels.
    """

    def __init__(
        self, data: Data, 
        predictions,
        similarity_func=None, w2v_model=None
    ):
        """
        Parameters:
            data: Data class object
            predictions: all predictions with index corresponding to original unlabeled_data
            similarity_func: function for similarity score, default cosine Word2Vec similarity
            w2v_model: the (pre)trained model to generate embedding vectors
        """
        assert len(data.unlabeled_labels) == len(predictions)

        self.data = data
        self.full_df = pd.DataFrame(
            {'predictions': predictions, 'truth': data.unlabeled_labels}
        )

        if similarity_func is None:
            if w2v_model is None:
                self.similarity_model = gensim_api.load("word2vec-google-news-300")
            else:
                self.similarity_model = w2v_model
            self.similarity_func = lambda u, v: w2v_cosine_similiarity(self.similarity_model, u, v)
        else:
            self.similarity_func = similarity_func
        
        self.existing_labels = set(data.labeled_labels).intersection(set(data.unlabeled_labels))
        self.removed_labels = set(data.unlabeled_labels) - set(data.labeled_labels)

        # existing labels: take existing labels from ground truth
        self.existing_idx, self.existing_pred, self.existing_label = zip(*[
            (idx, pred, truth) 
            for idx, (pred, truth) in enumerate(zip(self.full_df['predictions'], data.unlabeled_labels))
            if truth in self.existing_labels
        ])

        # pending label: to map
        self.pending_idx, self.pending_pred, self.pending_label = zip(*[
            (idx, pred, truth) 
            for idx, (pred, truth) in enumerate(zip(self.full_df['predictions'], data.unlabeled_labels))
            if pred not in self.existing_labels
        ])

        ## NOTE: The above 2 categories are NOT mutually exclusive.

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
    
    def auto_unconstrained_mapping(self):
        """
        Map each generated label to a removed label. Duplicates allowed.
        This is the naive mapping.
        """
        generated_classes = list(set(self.pending_pred) - set(self.existing_labels))
        similarity = pd.DataFrame([
            [self.similarity_func(p.lower(), t.lower()) for t in self.removed_labels] 
            for p in generated_classes
        ], index=generated_classes, columns=list(self.removed_labels))

        return similarity.dropna(axis=1).idxmax(axis=1).to_dict()
    
    def evaluate_full(self, mapping: dict):
        """
        General evaluation on full set of unlabeled data
        """
        mapped_table = self.full_df.assign(
            mapped=self.full_df['predictions'].apply(
                lambda s: mapping[s] if s not in self.existing_labels else s
            )
        )
        micro = f1_score(
            y_true=mapped_table['truth'], y_pred=mapped_table['mapped'], 
            average='micro'
        )
        macro = f1_score(
            y_true=mapped_table['truth'], y_pred=mapped_table['mapped'], 
            average='macro'
        )
        return {'micro_f1': round(micro, 3), 'macro_f1': round(macro, 3)}

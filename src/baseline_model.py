from util import *
from data import *
from base_modules import *
from evaluation import *
from datetime import datetime
import gensim.downloader as gensim_api


import warnings
warnings.filterwarnings('ignore')

class Baseline_Model:
    def __init__(self, data):
        self.data = data
        
    def run(self, w2v_config=None, w2v_model=None):
        if not os.path.exists('artifacts'):
            os.mkdir('artifacts')
        now = datetime.now().strftime('%Y-%m-%d')

        # Seed word generation: TF-IDF
        print('## Finding Seed Words')
        df_labeled = pd.DataFrame({'sentence': [
            ' '.join(doc) for doc in self.data.labeled_corpus
        ], 'label': self.data.labeled_labels})

        combined_text_by_topics = (
            df_labeled.groupby('label')['sentence']
            .apply(lambda doc: ' '.join(doc))
            .to_frame()
        )

        tfidf = Tfidf_Model()
        tfidf.fit_transform(combined_text_by_topics)
        seed_words = tfidf.get_top_dict(k=10)
        print()

        # Weakly supervised representation learning
        print('## Training or Learning Word2Vec Model')
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
        sim_ax = w2v.get_max_sim_distribution()
        sim_ax.get_figure().savefig(f'artifacts/{self.data.name}_sim_distribution_{now}.png')
        print()

        # get unconfident
        to_split = input("Insert the threshold for unconfident document: ")
        split_dict = w2v.confidence_split(float(to_split))
        unconfident_docs, unconfident_idx, unconfident_rep = split_dict['unconfident']
        confident_predictions = split_dict['confident']
        print("Number of Unconfident Documents:", len(unconfident_docs))

        if len(unconfident_docs) > 0:
            # clustering unconfident
            if len(unconfident_docs) > 1:
                gmm = Clustering_Model(method='gmm', vectors=unconfident_rep, vector_idx=unconfident_idx)
                gmm_results = gmm.fit_transform(n_classes=min(5, len(unconfident_docs)))
            else:
                gmm_results = [0]

            df_new_classes = pd.DataFrame({'sentence': [
                ' '.join(doc) for doc in unconfident_docs
            ], 'cluster_id': gmm_results}, index=unconfident_idx)
            combined_text_by_clusters = (
                df_new_classes.groupby('cluster_id')['sentence']
                .apply(lambda lst: ' '.join(lst)).to_frame()
            )
            
            # predict new class label
            tfidf_clusters = Tfidf_Model()
            tfidf_clusters.fit_transform(combined_text_by_clusters)
            k = 1 if self.data.name == 'testdata' else 6
            strict = True if self.data.name == 'testdata' else False
            cluster_labels = tfidf_clusters.get_top_dict(k=k, strict=strict)
            predict_cluster_word = {cluster_id: top[0] for cluster_id, top in cluster_labels.items()}
            self.predict_cluster_word = predict_cluster_word
            new_predictions = df_new_classes.assign(
                predictions=[predict_cluster_word[cid] for cid in gmm_results]
            )

            full_pred = pd.concat([
                new_predictions[['sentence', 'predictions']], 
                confident_predictions[['sentence', 'predictions']]
            ]).sort_index()
        else:
            full_pred = confident_predictions[['sentence', 'predictions']]

        return full_pred

    def evaluate(self, full_pred):
        eval_w2v_model = gensim_api.load('word2vec-google-news-300')
        e = Evaluation(self.data, full_pred['predictions'], w2v_model=eval_w2v_model)

        print("New Label Detection Metrics:")
        print(e.evaluate_new_label_metrics())
        print()

        print("Existing Label Performances:")
        print(e.evaluate_existing())
        print()

        if hasattr(self, 'predict_cluster_word'):
            print("Generated Class vs. Removed Labels:")
            print("Generated", set(self.predict_cluster_word.values()))
            print("Removed:", e.removed_labels)
            print()

            print("General Performance:")
            naive_map = e.auto_unconstrained_mapping()
            print(e.evaluate_full(naive_map))

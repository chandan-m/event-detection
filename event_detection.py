import pandas as pd
import logging
import yaml
import argparse
import warnings
from modules.event_detection.feature_matrix import build_matrix
from modules.event_detection.clustering import ClusteringAlgo
from modules.event_detection.evaluation import cluster_event_match

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ["wc_w2v_hierarchical", "wc_w2v_kmeans", "wc_glove_hierarchical", "wc_glove_kmeans"]
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model',
                    required=False,
                    choices=text_embeddings,
                    help="""
                    One or several text embeddings
                    """
                    )
parser.add_argument('--dataset',
                    required=False,
                    help="""
                    Path to the dataset
                    """
                    )
parser.add_argument('--threshold',
                    required=False
                    )


def main(args):
    with open("parameters.yaml", "r") as f:
        options = yaml.safe_load(f)
    params = options["event_detection"]
    for arg in args:
        if args[arg]:
            params[arg] = args[arg]
    logging.info("Clustering with {} model".format(params["model"]))
    run_model(**params)


def run_model(**params):
    thresholds = params.pop("threshold")
    X, data = build_matrix(**params)
    sub_data = None
    params["window"] = int(data.groupby("date").size().mean()//params["batch_size"]*params["batch_size"])
    logging.info("window size: {}".format(params["window"]))
    # threshold = params["threshold"]
    for threshold in thresholds:
        logging.info("threshold: {}".format(threshold))

        clustering = ClusteringAlgo(threshold=float(threshold), window_size=params["window"],
                                    batch_size=params["batch_size"])
        clustering.add_vectors(X)
        y_pred = clustering.incremental_clustering()

        data["pred"] = pd.Series(y_pred, dtype=data.label.dtype)
        data = data[data.pred != -1]
        data = data[data.groupby('pred').pred.transform('count') >= 5]
        p, r, f1 = cluster_event_match(data)
        stats = dict()
        stats.update({"model": params["model"], "t": threshold, "p": p, "r": r, "f1": f1})
        stats = pd.DataFrame(stats, index=[0])
        logging.info(stats[["t", "model", "p", "r", "f1"]].iloc[0])
        if params["save_results"]:
            try:
                results = pd.read_csv("output/evaluation_results.csv")
            except FileNotFoundError:
                results = pd.DataFrame()
            stats = results.append(stats, ignore_index=True)
            stats.to_csv("output/evaluation_results.csv", index=False)
            logging.info("Saved results to evaluation_results.csv")
        if threshold == 0.40:
            sub_data = data

    header = ["pred", "orig_tweet", "id"]
    if sub_data is not None:
        sub_data.to_csv(params["output"], columns=header, sep='\t')


def warn(*args, **kwargs):
    pass
warnings.warn = warn

if __name__ == '__main__':
    print("Event Detection - Module")
    args = vars(parser.parse_args())
    main(args)

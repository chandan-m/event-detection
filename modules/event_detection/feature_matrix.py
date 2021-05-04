import logging
import numpy as np
import os
from scipy.sparse import issparse, save_npz, load_npz
from modules.event_detection.load_dataset import load_dataset
from modules.event_detection.embeddings import WordClusteringVectorizer


logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ["wc_w2v_hierarchical", "wc_w2v_kmeans", "wc_glove_hierarchical", "wc_glove_kmeans"]


def save_matrix(X, **args):
    path = os.path.join("modules", "event_detection", "saved", args["model"], "saved_matrix")
    if issparse(X):
        save_npz(path, X)
    else:
        np.save(path, X)


def load_matrix(**args):
    path = os.path.join("modules", "event_detection", "saved", args["model"], "saved_matrix")
    os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
    for suffix in [".npy", ".npz"]:
        if os.path.exists(path + suffix):
            return np.load(path + suffix) if suffix == ".npy" else load_npz(path + suffix)


def build_matrix(**args):
    data = load_dataset(args["dataset"])
    if args["load_saved_matrix"]:
        X = load_matrix(**args)
        if X is not None:
            logging.info("Matrix already stored")
            return X, data

    logging.info("Generating Data Matrix")

    # Word Clustering Vectorization
    vectorizer = WordClusteringVectorizer(model=args["model"])
    X = vectorizer.compute_vectors(data)

    if args["save_matrix"]:
        save_matrix(X, **args)

    return X, data

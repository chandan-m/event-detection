import logging
import pandas as pd
import time


def cluster_event_match(data):
    data = data[data.label.notna()]
    logging.info("{} labels, {} preds".format(len(data.label.unique()), len(data.pred.unique())))
    t0 = time.time()

    # Group data by actual label and predicted label...
    # ... and store count of each group as  "a"
    # a - True Positive
    # b - False Positive
    # c - False Negative
    match = data.groupby(["label", "pred"], sort=False).size().reset_index(name="a")
    b, c = [], []
    for idx, row in match.iterrows():
        b_ = ((data["label"] != row["label"]) & (data["pred"] == row["pred"]))
        b.append(b_.sum())
        c_ = ((data["label"] == row["label"]) & (data["pred"] != row["pred"]))
        c.append(c_.sum())
    logging.info("Event Cluster Match - Time Taken: {} seconds".format(time.time() - t0))

    match["b"] = pd.Series(b)
    match["c"] = pd.Series(c)
    # Recall = TP / (TP + FN)
    match["r"] = match["a"] / (match["a"] + match["c"])
    # Precision = TP / (TP + FP)
    match["p"] = match["a"] / (match["a"] + match["b"])
    # F1 Score = (2 * Recall * Precision) / (Recall + Precision)
    match["f1"] = 2 * match["r"] * match["p"] / (match["r"] + match["p"])
    match = match.sort_values("f1", ascending=False)
    macro_average_f1 = match.drop_duplicates("label").f1.mean()
    macro_average_precision = match.drop_duplicates("label").p.mean()
    macro_average_recall = match.drop_duplicates("label").r.mean()
    return macro_average_precision, macro_average_recall, macro_average_f1

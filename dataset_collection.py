import yaml
from modules.dataset_collection.rehydrate_tweets import rehydrate_tweets, format_tweet
import logging

import pandas as pd
import csv

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


def main():
    with open("parameters.yaml", "r") as f:
        options = yaml.safe_load(f)
    params = options["dataset_collection"]
    tweet_ids_path = params.pop("tweet_ids_path")
    rehydrated_dataset = params.pop("output")
    labeled_data = pd.read_csv(tweet_ids_path, sep="\t", header=None, names=["label", "id"],
                               dtype={"id": str}
                               ).drop_duplicates()
    ids = labeled_data.id.tolist()
    tweets = rehydrate_tweets(params, ids, jsondump=True)
    complete_data = pd.DataFrame([format_tweet(row) for row in tweets])
    complete_data = labeled_data.merge(complete_data, on="id", how="left")
    complete_data = complete_data[complete_data.text.notna()]
    complete_data["label"] = complete_data["label"]
    complete_data["text"] = complete_data["text"].str.replace("\t", " ").str.replace("\n", " ").str.replace("\r", " ")
    complete_data.to_csv(rehydrated_dataset, sep="\t", index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()

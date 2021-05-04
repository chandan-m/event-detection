import json
import time
from twython import Twython, TwythonRateLimitError
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)

def rehydrate_tweets(params, tweet_id_list, jsondump=False):
    logging.info("Starting calls to Twitter API. This may take some time.")
    twython_obj = Twython(**params)
    tweets = []
    id_count = 0
    while True:
        try:
            batch = twython_obj.lookup_status(
                id=tweet_id_list[id_count:id_count+100],
                include_entities=True,
                tweet_mode="extended")
        except TwythonRateLimitError:
            reset_time = float(twython_obj.get_lastfunction_header("x-rate-limit-reset"))
            delta = round(int(reset_time) - time.time(), 0)
            logging.warning("Twitter rate limit reached, sleeping {} seconds".format(delta + 1))
            time.sleep(delta + 1)
            continue
        if len(batch) == 0:
            break
        if jsondump:
            with open("data/event_2018.json", "w") as f:
                for tweet in batch:
                    json_str = json.dumps(tweet) + "\n"
                    f.write(json_str)
        tweets += batch
        id_count += 100
        logging.info(" ... {} / {} tweets retrieved so far".format(id_count, len(tweet_id_list)))

    # Check to see if we didn't get all of the requested tweets back
    if len(tweet_id_list) != len(tweets):
        logging.info("{}% of ids collected. Some tweets/accounts may have been deleted.".format(
            round(100*len(tweets)/len(tweet_id_list), 0))
        )

    return tweets


def format_tweet(row):
    tweet_dict = {"text": row["full_text"], "id": row["id_str"], "created_at": row["created_at"]}
    if "extended_entities" in row and "media" in row["extended_entities"]:
        tweet_dict["url_image"] = row["extended_entities"]["media"][0]["media_url"]
    return tweet_dict

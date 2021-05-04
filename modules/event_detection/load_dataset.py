import pandas as pd
import csv
from datetime import datetime
import re
from unidecode import unidecode

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"
STANDARD_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def find_date_created_at(created_at):
    if "+0000" in created_at:
        d = datetime.strptime(created_at, TWITTER_DATE_FORMAT)
    else:
        d = datetime.strptime(created_at, STANDARD_DATE_FORMAT)
    return d.strftime("%Y%m%d"), d.strftime("%H:%M:%S")


def remove_repeted_characters(expr):
    # Remove repeated characters in text like: oooook -> ok
    string_not_repeted = ""
    for item in re.findall(r"((.)\2*)", expr):
        if len(item[0]) <= 3:
            string_not_repeted += item[0]
        else:
            string_not_repeted += item[0][:3]
    return string_not_repeted


def hashtag_split(expr):
    # Split hashtags based on camel case.
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', expr)
    return " ".join([m.group(0) for m in matches])


def format_text(text, **args):
    # Remove URLs from Tweet text.
    text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)
    if args["remove_mentions"]:
        text = re.sub(r"@\S+", '', text, flags=re.MULTILINE)
    # Convert to ASCII characters
    if args["unidecode"]:
        text = unidecode(text)
    new_text = []
    for word in re.split(r"[' ]", text):
        if len(word) < 5 or not word.isdigit():
            if word.startswith("#") and args["hashtag_split"]:
                new_text.append(hashtag_split(word[1:]))
            else:
                new_text.append(word)
    text = remove_repeted_characters(" ".join(new_text))
    if args["lower"]:
        text = text.lower()
    return text


def apply_formatting(data):
    data["orig_tweet"] = data["text"]
    data.text = data.text.apply(format_text,
                                remove_mentions=True,
                                unidecode=True,
                                lower=True,
                                hashtag_split=True
                                )
    return data


def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path,
                       sep="\t",
                       quoting=csv.QUOTE_ALL,
                       dtype={"id": str, "label": float, "created_at": str, "text": str}
                       )
    data.text = data.text.fillna("")
    data = data[data.label.notna()]
    data["date"], data["time"] = zip(*data["created_at"].apply(find_date_created_at))
    data = data.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    return apply_formatting(data)

print("Importing Modules ...")
import csv
import re
import string
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger
import sqlite3
from sqlite3 import Error
import json

print("Completed Importing Modules.")

max_words_in_sentence = 30
max_tweets_inSummary = 50
file_path = "output/"
file_name = "event_detection_output.tsv"
db_name = "analysis_results.db"
csv_file_name = file_path + "analysis_results.csv"
id_column_number = 1
tweet_column_number = 2
create_id_tweet_table = True
final_d = {}
tsv_file = open(file_path + file_name, encoding="utf8")
read_tsv = csv.reader(tsv_file, delimiter="\t")

print("Loading data...")
l = []
first = True
for i in read_tsv:
    if (first):
        first = False
        continue
    if (str(i[id_column_number]) != "-1" and str(i[id_column_number]) != "-1.0"):
        l.append(i)
print("Loading data completed.")

print("Cleaning tweets...")
d = {}
for x in l:
    id = x[id_column_number]
    actual_text = x[tweet_column_number]
    formatted_article_text = re.sub(r'http\S+', ' ', actual_text)
    formatted_article_text = re.sub('[^a-zA-Z0-9#@]', ' ', formatted_article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text = formatted_article_text.lower()

    actual_text = re.sub(r'http\S+', ' ', actual_text)
    actual_text = re.sub(r'\s+', ' ', actual_text)
    text_list = [formatted_article_text, actual_text]
    if id not in d:
        d[id] = []
    d[id].append(text_list)
print("Completed cleaning tweets.")

print("Generating Event Cluster Representation")
stopwords = nltk.corpus.stopwords.words('english')

word_freq = {}
for i in d:
    list_of_sentences = d[i]
    word_freq[i] = {}
    for pair in list_of_sentences:
        clean_text = pair[0]
        all_words = clean_text.split()
        for word in all_words:
            if word not in stopwords:
                if word not in word_freq[i]:
                    word_freq[i][word] = 1
                else:
                    word_freq[i][word] += 1

for g in word_freq:
    high = max(word_freq[g].values())
    for word in word_freq[g]:
        word_freq[g][word] = round(word_freq[g][word] / high, 2)

result_d = {}
for i in d:
    result_d[i] = {}
    scores = word_freq[i]
    for pair in d[i]:
        sentence_score = 0
        text = pair[1].lower()
        words = nltk.word_tokenize(text)
        if (len(words) < max_words_in_sentence):
            for w in words:
                if (w in scores):
                    sentence_score += scores[w]
        result_d[i][pair[1]] = sentence_score

for i in result_d:
    f = sorted(result_d[i], key=result_d[i].get, reverse=True)
    res = []
    [res.append(x) for x in f if x not in res]
    f = res
    final_list = f[0:max_tweets_inSummary]
    if i not in final_d:
        final_d[i] = {}
    final_d[i]["Tweets"] = final_list

print("Completed event representations.")


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def get_sentiment(classifier, custom_tweet):
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    s = classifier.classify(dict([token, True] for token in custom_tokens))
    return s


print("Collecting data for training sentiment analysis model..")
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

stop_words = stopwords

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

print("Training the model for sentiment analysis..")
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10))

for i in final_d:
    tweet = final_d[i]["Tweets"][0]
    sentiment = get_sentiment(classifier, tweet)
    final_d[i]["Sentiment"] = sentiment
    final_d[i]["Positive"] = 0
    final_d[i]["Negative"] = 0

print("Performing Sentiment Analysis")
for x in l:
    cid = x[id_column_number]
    tweet = x[tweet_column_number]
    sentiment = get_sentiment(classifier, tweet)
    if sentiment == "Positive":
        final_d[cid]["Positive"] += 1
    elif sentiment == "Negative":
        final_d[cid]["Negative"] += 1
    else:
        print(sentiment, x)
print("Completed Performing Sentiment Analysis")

print("Performing Entity extraction..")
print("Load Tagger..")
tagger = SequenceTagger.load('ner')

for x in final_d:
    print("Performing Entity extraction for Cluster number: " + str(x))
    listTweet = final_d[x]["Tweets"]
    listTweet = listTweet[0:3]
    org = set()
    per = set()
    loc = set()
    oth = set()
    for i in listTweet:
        sentence = Sentence(i.upper())
        tagger.predict(sentence)
        g = sentence.to_dict(tag_type='ner')
        lg = g['entities']
        for ent in lg:
            if 'labels' in ent:
                q = ent['labels'][0].to_dict()['value']
                if q == 'PER':
                    per.add(ent['text'])
                elif q == 'ORG':
                    org.add(ent['text'])
                elif q == 'LOC':
                    loc.add(ent['text'])
                else:
                    oth.add(ent['text'])
    final_d[x]["Org"] = list(org)
    final_d[x]["Per"] = list(per)
    final_d[x]["Loc"] = list(loc)
    final_d[x]["Oth"] = list(oth)

print("Completed Performing Entity extraction..")

print("Writing to database")

db = file_path + db_name


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def save_output_db(conn, c_id, sentiment, tweet1, tweet2, org, loc, per, oth, n, p):
    org_str = ','.join(org)
    loc_str = ','.join(loc)
    per_str = ','.join(per)
    oth_str = ','.join(oth)
    conn.execute(
        "INSERT INTO results(c_id,sentiment,tweet1,tweet2,org,loc,per,oth,negative,positive) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(c_id), str(sentiment), str(tweet1), str(tweet2), org_str, loc_str, per_str, oth_str, n, p))
    conn.commit()
    return True


def id_tweet_table(conn, c_id, tweet):
    conn.execute("INSERT INTO tweets(c_id,tweet) VALUES (?,?)",
                 (c_id, tweet))
    conn.commit()
    return True


drop_results_table = """ DROP TABLE IF EXISTS results;
"""

drop_results_table_2 = """ DROP TABLE IF EXISTS tweets;
"""

sql_create_results_table = """ CREATE TABLE results (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    c_id text,
                                                    sentiment text,
                                                    tweet1 text,
                                                    tweet2 text,
                                                    org text,
                                                    loc text,
                                                    per text,
                                                    oth text,
                                                    negative INTEGER,
                                                    positive INTEGER
                                                ); """
sql_create_results_table_2 = """ CREATE TABLE tweets (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    c_id text,
                                                    tweet text
                                                ); """
conn = create_connection(db)
if conn is not None:
    create_table(conn, drop_results_table)
    create_table(conn, drop_results_table_2)
    create_table(conn, sql_create_results_table)
    create_table(conn, sql_create_results_table_2)
else:
    print("Error! cannot create the database connection.")

if (create_id_tweet_table):
    print("Writing into id_tweet_table")
    for i in final_d:
        print("Writing into db file for cluster number: " + str(i))
        for x in final_d[i]["Tweets"]:
            id_tweet_table(conn, str(i), str(x))

row_list = [["c_id", "sentiment", "tweet1", "tweet2", "org", "loc", "per", "oth", "negative", "positive"]]

print("Final output")
for i in final_d:
    print(i)
    print("Negative Tweets count: " + str(final_d[i]["Negative"]) + "\t" + "Postive Tweets count: " + str(
        final_d[i]["Positive"]))
    print("Top 2 Tweets:")
    h_count = 0
    for x in final_d[i]["Tweets"]:
        print(x)
        h_count += 1
        if (h_count == 2):
            break
    print("Entities")
    print("Organisations: ", final_d[i]["Org"])
    print("Persons: ", final_d[i]["Per"])
    print("Locations: ", final_d[i]["Loc"])
    print("Others: ", final_d[i]["Oth"])
    print()
    print()
    if len(final_d[i]["Tweets"]) > 1:
        save_output_db(conn, i, final_d[i]["Sentiment"], final_d[i]["Tweets"][0], final_d[i]["Tweets"][1],
                       final_d[i]["Org"], final_d[i]["Loc"], final_d[i]["Per"], final_d[i]["Oth"],
                       final_d[i]["Negative"], final_d[i]["Positive"])
        row_list.append(
            [str(i), str(final_d[i]["Sentiment"]), str(final_d[i]["Tweets"][0]), str(final_d[i]["Tweets"][1]),
             ','.join(final_d[i]["Org"]), ','.join(final_d[i]["Loc"]), ','.join(final_d[i]["Per"]),
             ','.join(final_d[i]["Oth"]), final_d[i]["Negative"], final_d[i]["Positive"]])

print("Writing into csv file")
with open(csv_file_name, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

print("Completed event analysis")
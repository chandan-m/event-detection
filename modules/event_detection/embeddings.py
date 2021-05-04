import re

import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import modules.event_detection.stop_words as stop_words
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import scipy.cluster.hierarchy as sch
from scipy import sparse
import fastcluster
from sklearn import preprocessing
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

__all__ = ["WordClusteringVectorizer"]

class WordClusteringVectorizer:

    def __init__(self, model):
        self.name = "WordClusteringVectorizer"
        self.model = model
        self.data = None
        self.stop_words = stop_words.STOP_WORDS_EN
        self.vocab_words = None
        self.vocab_words_df = None
        self.vocab_words_vectors = None
        self.data_preprocessed = None
        self.w2v_model = None
        self.idx = None
        self.df = None
        self.tweet_vectors = None

    def hasDigits(self, s):
        return any(48 <= ord(char) <= 57 for char in s)

    def getVocab(self):
        vectorizer = CountVectorizer(binary=True, min_df=10, stop_words=self.stop_words)
        count_vectors = vectorizer.fit_transform(self.data)
        self.vocab_words = vectorizer.get_feature_names()
        self.vocab_words_df = count_vectors.toarray().sum(axis=0).tolist()  # Only if count_vector is binary

    def filterVocab(self):
        vocab_words = list()
        vocab_words_df = list()
        vocab_words_vectors = list()
        print("Before Filtering:", len(self.vocab_words), len(self.vocab_words_df), len(self.vocab_words_vectors))
        for i in range(len(self.vocab_words)):
            if ((not self.hasDigits(self.vocab_words[i])) and (self.vocab_words_vectors[i] is not None)):
                vocab_words.append(self.vocab_words[i])
                vocab_words_df.append(self.vocab_words_df[i])
                vocab_words_vectors.append(self.vocab_words_vectors[i])
        self.vocab_words = vocab_words
        self.vocab_words_df = vocab_words_df
        self.vocab_words_vectors = vocab_words_vectors
        print("After Filtering:", len(self.vocab_words), len(self.vocab_words_df), len(self.vocab_words_vectors))

    def preprocess_util(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = list(text.split())
        new_tokens = list()
        for token in tokens:
            if len(token) <= 1:
                continue
            new_tokens.append(token)
        return new_tokens

    def preprocess(self):
        data_preprocessed = list()
        for tweet in self.data:
            data_preprocessed.append(self.preprocess_util(tweet))
        self.data_preprocessed = data_preprocessed

    def getWordEmbeddings_w2v(self):
        w2v = Word2Vec(self.data_preprocessed, size=200, window=5, min_count=10,
                       negative=15, iter=10, workers=multiprocessing.cpu_count(), sg=1)
        word_vectors_mapping = w2v.wv
        vocab_words_vectors = list()
        for word in self.vocab_words:
            if word in word_vectors_mapping:
                vocab_words_vectors.append(word_vectors_mapping[word])
            else:
                vocab_words_vectors.append(None)
        self.vocab_words_vectors = vocab_words_vectors
        self.w2v_model = word_vectors_mapping

    def getWordEmbeddings_glove(self):
        glove_embedding = WordEmbeddings('glove')
        vocab_words_vectors = list()
        for word in self.vocab_words:
            sentence = Sentence(word)
            glove_embedding.embed(sentence)
            if len(sentence) > 0:
                vocab_words_vectors.append(sentence[0].embedding.numpy())
            else:
                vocab_words_vectors.append(None)
        self.vocab_words_vectors = vocab_words_vectors

    def cluster_words_vectors_k_means(self):
        Kmean = KMeans(n_clusters=1000, init='k-means++');
        idx = Kmean.fit_predict(self.vocab_words_vectors)
        self.idx = idx

    def cluster_words_vectors_hierarchical(self):
        input_matrix = np.matrix(self.vocab_words_vectors).astype('float')
        matrix_scaled = preprocessing.scale(input_matrix)
        matrix_normalized = preprocessing.normalize(matrix_scaled)
        distance_matrix = pairwise_distances(matrix_normalized, metric='cosine')
        linkage_matrix = fastcluster.linkage(distance_matrix, method='average')
        distance_ratio = 5
        max_distance = distance_matrix.max()
        distance_threshold = distance_ratio * max_distance
        flat_clusters = sch.fcluster(linkage_matrix, distance_threshold, 'distance')
        self.idx = flat_clusters - 1

    def get_tweet_vectors(self):
        vectorizer = CountVectorizer(binary=True, vocabulary=self.vocab_words)
        count_vectors = vectorizer.fit_transform(self.data)
        count_vectors = count_vectors.toarray()
        dim = max(self.idx) + 1
        bin_vec = np.zeros((len(self.data), dim), dtype=int)
        df = np.zeros(dim, dtype=int)
        for i in range(count_vectors.shape[0]):
            for j in range(count_vectors.shape[1]):
                if (count_vectors[i][j] >= 1):
                    bin_vec[i][self.idx[j]] += 1
                    df[self.idx[j]] += self.vocab_words_df[j]
        bin_vec_sparse = sparse.csr_matrix(np.matrix(bin_vec).astype('float'))
        self.tweet_vectors = bin_vec_sparse

    def process(self, data):
        self.data = data["text"].tolist()
        self.preprocess()
        self.getVocab()
        if "w2v" in self.model:
            self.getWordEmbeddings_w2v()
        elif "glove" in self.model:
            self.getWordEmbeddings_glove()
        self.filterVocab()
        if "hierarchical" in self.model:
            self.cluster_words_vectors_hierarchical()
        elif "kmeans" in self.model:
            self.cluster_words_vectors_k_means()

    def compute_vectors(self, data):
        self.process(data)
        print("Finished clustering Words")
        self.get_tweet_vectors()
        return self.tweet_vectors

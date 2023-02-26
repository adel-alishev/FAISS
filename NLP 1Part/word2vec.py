import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
from gensim.models import Word2Vec
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

df = pd.read_csv('brand_tweets.csv', sep=',',encoding='utf8')
df.drop(df[df.tweet_text.isnull()].index, inplace=True)

def tokenized_text(raw_text:str):
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation)]
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_tokens = [i for i in tokens if (i not in stop_words)]
    return filtered_tokens

tokenized_tweets = df.tweet_text.apply(tokenized_text)

df = df.assign(tokenized = tokenized_tweets)
texts = df.tokenized.values

model = Word2Vec(
    texts,
    vector_size=10,
    window=7,
    min_count=2,
    workers=4,
    epochs=10,
    sg=0)
print(model.wv.get_vector('iphone'))
print(model.wv.most_similar('iphone'))
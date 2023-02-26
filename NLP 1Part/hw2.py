import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

df = pd.read_csv('brand_tweets_valid.csv', sep=',',encoding='utf8')
df.drop(df[df.tweet_text.isnull()].index, inplace=True)
print(df.shape)

def tokenized_text(raw_text:str):
    #nltk.download('punkt', download_dir='.')
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation)]
    #nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_tokens = [i for i in tokens if (i not in stop_words)]
    #print(filtered_tokens)
    return filtered_tokens

vectorizer = CountVectorizer(tokenizer=tokenized_text)
document_matrix = vectorizer.fit_transform(df.tweet_text.values)
#print(document_matrix)

source_tweet_index = 14
#print(df.tweet_text.values[source_tweet_index])
tweet_distance = 1-pairwise_distances(document_matrix, metric='cosine')

sorted_similarity = np.argsort(-tweet_distance[source_tweet_index, :])
#print(sorted_similarity)
print(df.iloc[source_tweet_index].tweet_text)
print('------------------------------')
print(df.iloc[sorted_similarity[1]].tweet_text)
print('------------------------------')
print(df.iloc[sorted_similarity[2]].tweet_text)
print('------------------------------')
print(df.iloc[sorted_similarity[3]].tweet_text)
print('------------------------------')
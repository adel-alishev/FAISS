from matplotlib import pyplot as plt
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
import seaborn as sns

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
def hist_n_simil(index, alpha=0.5):
    tweet_distance = 1 - pairwise_distances(document_matrix, metric='cosine')
    sorted_similarity = np.argsort(-tweet_distance[index, :])
    cond = np.argwhere(tweet_distance[index,:]>alpha)
    x = cond.reshape(1,len(cond))[0]
    y = tweet_distance[index,cond].reshape(1,len(cond))[0]
    #print(x,y)
    sns.barplot(x = x, y=y)
    plt.show()
    print(df.iloc[index].tweet_text)
    print('------------------------------')
    for i in range(1,len(cond)):
        print(df.iloc[sorted_similarity[i]].tweet_text)
        print('------------------------------')
hist_n_simil(15, alpha=0.45)
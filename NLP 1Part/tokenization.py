import nltk
import string

import pandas as pd

df = pd.read_csv('brand_tweets.csv', sep=',',encoding='utf8')
print(df.shape)

df.drop(df[df.tweet_text.isnull()].index, inplace=True)
print(df.shape)

sample_str = df.tweet_text.values[0]
#print('========Исходный текст:==============='+sample_str)

#nltk.download('punkt', download_dir='.')
tokenized_str = nltk.word_tokenize(sample_str)
#print('========Токенизированный текст:======='+str(tokenized_str))

tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation)]
#print(tokens)

#nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
#print(stop_words)

filtered_tokens = [i for i in tokens if (i not in stop_words)]
#print(filtered_tokens)

def tokenized_text(raw_text:str):
    #nltk.download('punkt', download_dir='.')
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation)]
    #nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_tokens = [i for i in tokens if (i not in stop_words)]
    #print(filtered_tokens)
    return filtered_tokens

tokenized_text(sample_str)

tokenized_tweets = df.tweet_text.apply(tokenized_text)
#print(tokenized_tweets.head(7))

df = df.assign(tokenized = tokenized_tweets)
print(df.head(7))
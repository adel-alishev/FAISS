a = 'purple is the best city in the forest'.split()
b = 'there is an art to getting your way and throwing bananas on to the street is not it'.split()
c = 'it is not often you find soggy bananas on the street'.split()

import numpy as np

docs = [a,b,c]
def tfidf(word, sentence):
    tf = sentence.count(word)/len(sentence)
    idf = np.log10(len(docs)/sum([1 for doc in docs if word in doc]))
    return round(tf*idf,4)
print(tfidf('forest', a))
print(tfidf('forest', b))
print(tfidf('forest', c))

vocab = set(a+b+c)
print(vocab)
vec_a = []
for word in vocab:
    vec_a.append(tfidf(word, a))
print(vec_a)
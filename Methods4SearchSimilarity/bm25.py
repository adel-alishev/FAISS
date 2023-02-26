a = 'purple is the best city in the forest'.split()
b = 'there is an art to getting your way and throwing bananas on to the street is not it'.split()
c = 'it is not often you find soggy bananas on the street'.split()
d = 'green should have smelled more tranquil but somehow it just tasted rotten'.split()
e = 'joyce enjoyed eating pancakes with ketchup'.split()
f = 'as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled'.split()

docs = [a,b,c,d,e,f]

import numpy as np
def tfidf(word, sentence):
    tf = sentence.count(word)/len(sentence)
    idf = np.log10(len(docs)/sum([1 for doc in docs if word in doc]))
    return round(tf*idf,4)
avgd1 = sum(len(sentence) for sentence in [a,b,c,d,e,f])/len(docs)
N = len(docs)

def bm25(word, sentence, k=1.2, b=0.75):
    freq = sentence.count(word)
    tf = (freq*(k+1))/(freq+k*(1-b+b*len(sentence)/avgd1))
    N_q = sum([1 for doc in docs if word in doc])
    idf = np.log10(((N-N_q+0.5)/(N_q+0.5))+1)
    return round(tf*idf,4)
print(bm25('purple',a))
import numpy as np
dim = 512  # рассмотрим произвольные векторы размерности 512
nb = 10000  # количество векторов в индексе
nq = 5 # количество векторов в выборке для поиска
np.random.seed(228)
vectors = np.random.random((nb, dim)).astype('float32')
query = np.random.random((nq, dim)).astype('float32')

import faiss
index = faiss.IndexFlatL2(dim)
print(index.ntotal)  # пока индекс пустой
index.add(vectors)
print(index.ntotal)  # теперь в нем 10 000 векторов

topn = 7
D, I = index.search(vectors[:5], topn)  # Возвращает результат: Distances, Indices
print(I)
print(D)

D, I = index.search(query, topn)
print(I)
print(D)

faiss.write_index(index, "flat.index")
index = faiss.read_index("flat.index")
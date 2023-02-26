import numpy as np
a = 'his thought process was on so many levels that he gave himself a phobia of heights'
b = 'there is an art to getting your way and throwing bananas on to the street is not it'
c = 'it is not often you find soggy bananas on the street'
def leven(a,b):
    a = f' {a}'
    b = f' {b}'
    lev = np.zeros((len(a), len(b)))

    for i in range(len(a)):
        for j in range(len(b)):
            if min([i,j]) == 0:
                lev[i,j] = max([i,j])
            else:
                x = lev[i-1,j]
                y = lev[i, j-1]
                z = lev[i-1, j-1]
                lev[i,j] = min([x,y,z])
                if a[i] != b[j]:
                    lev[i,j] += 1
    return lev, lev[-1, -1]
print(leven('Levenshtein', 'Livinshten'))
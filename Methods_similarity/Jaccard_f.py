a = 'his thought process was on so many levels that he gave himself a phobia of heights'
b = 'there is an art to getting your way and throwing bananas on to the street is not it'
c = 'it is not often you find soggy bananas on the street'
def jaccard(x:str, y:str):
    x = set(x.split())
    y = set(y.split())
    shared = x.intersection(y)
    print(shared)
    union = x.union(y)
    #print(union)
    return len(shared)/len(union)

test = jaccard(b,c)
print(test)


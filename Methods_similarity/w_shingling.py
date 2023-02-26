a = 'his thought process was on so many levels that he gave himself a phobia of heights'
b = 'there is an art to getting your way and throwing bananas on to the street is not it'
c = 'it is not often you find soggy bananas on the street'

print([a[i] for i in range(len(a.split()))])
a = a.split()
print([a[i] for i in range(len(a))])
print([[a[i], a[i+1]] for i in range(len(a)-1)])
print(set([' '.join([a[i], a[i+1]]) for i in range(len(a)-1)]))
import numpy as np
a = open('tmp.csv').readlines()
b = map(lambda x: x.replace('(','').replace(')','').split(','), a)
c = list(b)
cc = np.array(c)
d = np.array(list(map(lambda y: list(map(lambda x: abs(complex(x)), y)), cc)))
print(d.argmax(1))

# coding: utf-8
import pickle
import numpy as np

for filename in ['black.n_visited', 'white.n_visited']:
    n = pickle.load(open(filename,'rb'))
    for i,v in n.items():
        if v > 100:
            v = 10*np.sqrt(v)
        n[i] = v
    pickle.dump(n, open(filename,'wb'))



import pickle
import matplotlib.pyplot as plt

data = pickle.load(open('cachehigh', 'rb'))

print("%d data imported"%len(data))

n_stones = [len(k[0]) + len(k[1]) for k in data.keys()]

plt.hist(n_stones, max(n_stones) - min(n_stones))

plt.savefig('nstone_distri.pdf')

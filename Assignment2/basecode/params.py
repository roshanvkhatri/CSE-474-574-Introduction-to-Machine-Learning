import pickle

with open("params.pickle", 'rb') as f:
    x = pickle.load(f)

print(x)
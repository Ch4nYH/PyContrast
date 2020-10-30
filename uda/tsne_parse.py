from sklearn import TSNE
import numpy as np
import pickle
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
features = pickle.load(open('features.pkl', 'rb'))
low_dim_embs = tsne.fit_transform(features)
target = np.array([1,2,3,4]).repeat(features.shape[0] // 4, 1)


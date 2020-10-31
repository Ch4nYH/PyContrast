from sklearn.manifold import TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
colors = ['red', 'blue', 'green', 'yellow']
def plot_with_labels(lowDWeights, labels, l):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        plt.scatter(x, y, color = colors[s - 1])
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')
    plt.show()
    plt.savefig("{}.jpg".format(l))


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
features = pickle.load(open('features.pkl', 'rb'))
low_dim_embs = tsne.fit_transform(features)
target = np.array([1,2,3,4]).repeat(features.shape[0] // 4)
for i in range(1, 5):
    plot_with_labels(low_dim_embs[target==i,:], target[target==i], i)


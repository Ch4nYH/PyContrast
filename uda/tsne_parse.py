from sklearn.manifold import TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')
    plt.show()
    plt.savefig("1.jpg")

    
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
features = pickle.load(open('features.pkl', 'rb'))
low_dim_embs = tsne.fit_transform(features)
target = np.array([1,2,3,4]).repeat(features.shape[0] // 4)
plot_with_labels(low_dim_embs, target)


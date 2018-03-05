import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_datasets():
    iris = datasets.load_iris()
    x_data, y_label = iris.data, iris.target
    x_data = (x_data - np.mean(x_data, axis=0)) / np.sqrt(np.var(x_data))

    n_samples, n_features = np.shape(x_data)
    idx = list(range(n_samples))
    random.shuffle(idx)
    x_data, y_label = x_data[idx], y_label[idx]

    return x_data, y_label


def lda_1():
    x_data, y_label = load_datasets()
    model = LinearDiscriminantAnalysis(solver='eigen')
    model.fit(x_data, y_label)


def lda_3(ax):
    x_data, y_label = load_datasets()
    n_samples, n_features = np.shape(x_data)
    classes = np.unique(y_label)

    u = np.mean(x_data, axis=0)

    s_w, s_b = np.zeros(shape=[n_features, n_features]), np.zeros(shape=[n_features, n_features])

    for c in classes:
        x_cluster = x_data[np.equal(y_label, c)]
        u_cluster = np.mean(x_cluster, axis=0)
        n_cluster = len(x_cluster)

        s_w += n_cluster * np.matmul((x_cluster - u_cluster).T, x_cluster - u_cluster)
        s_b += np.matmul(np.expand_dims(u_cluster - u, axis=1), np.expand_dims(u_cluster - u, axis=0))

    lamb, p = np.linalg.eigh(np.linalg.inv(s_w) * s_b)
    w = p[:, :3]

    x_transformed = np.matmul(x_data, w)
    colors = [('red', 'green', 'blue')[yy] for yy in y_label]
    ax.scatter(x_transformed[:, 0], x_transformed[:, 1], x_transformed[:, 2], c=colors)
    ax.set_title('decomposition using lda')


def pca(ax):
    x_data, y_label = load_datasets()
    model = decomposition.PCA(n_components=3)
    x_transformed = model.fit_transform(x_data)

    colors = [('red', 'green', 'blue')[yy] for yy in y_label]
    ax.scatter(x_transformed[:, 0], x_transformed[:, 1], x_transformed[:, 2], c=colors)
    ax.set_title('decomposition using pca')


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    lda_3(ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    pca(ax2)
    plt.show()

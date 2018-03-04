import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition


def load_datasets():
    iris = datasets.load_iris()
    x_data, y_label = iris.data, iris.target
    x_data = (x_data - np.mean(x_data, axis=0)) / np.sqrt(np.var(x_data))

    n_samples, n_features = np.shape(x_data)
    idx = list(range(n_samples))
    random.shuffle(idx)
    x_data, y_label = x_data[idx], y_label[idx]

    return x_data, y_label


def lda_3(ax):
    x_data, y_label = load_datasets()
    n_samples, n_features = np.shape(x_data)
    n_classes = np.max(y_label)

    u = np.mean(x_data, axis=0)

    s_w, s_b = np.zeros(shape=[n_features, n_features]), np.zeros(shape=[n_features, n_features])

    for i in range(n_classes):
        x_scatter = x_data[np.equal(y_label, i)]
        u_scatter = np.mean(x_scatter, axis=0)
        n_scatter = len(x_scatter)

        s_w += n_scatter * np.matmul((x_scatter - u_scatter).T, x_scatter - u_scatter)
        s_b += np.matmul(np.expand_dims(u_scatter - u, axis=1), np.expand_dims(u_scatter - u, axis=0))

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

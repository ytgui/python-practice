import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition, preprocessing


def pca_1(ax):
    x_data = np.loadtxt('x_data.csv')
    y_label = np.loadtxt('y_label.csv', dtype=np.int)
    print('original shape:', x_data.shape)

    transform = decomposition.PCA(n_components=3)
    x_transformed = transform.fit_transform(x_data)
    print('transformed shape:', x_data.shape)

    color = ['red', 'green', 'blue', 'yellow']
    for x, y in zip(x_transformed, y_label):
        ax.scatter(x[0], x[1], x[2], c=color[y])
    ax.set_title('pca by scikit-learn')


def pca_2(ax):
    # https://en.wikipedia.org/wiki/Principal_component_analysis
    x_data = np.loadtxt('x_data.csv', dtype=np.float64)
    y_label = np.loadtxt('y_label.csv', dtype=np.int)
    print('original shape:', x_data.shape)

    # 1. standardize data
    x1 = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
    x2 = preprocessing.StandardScaler().fit_transform(x_data)
    assert np.allclose(x1, x2)  # check
    x_data = x1

    # 2. compute cov matrix
    feature_cov = np.cov(x_data, rowvar=False)

    # 3. compute eig matrix
    lamb, P = np.linalg.eigh(feature_cov)
    idx = np.argsort(lamb)[::-1]
    lamb, P = lamb[idx], P[:, idx]

    # 4. keep major feature
    P = P[:, :3]
    x_transformed = np.matmul(P.T, x_data.T).T

    color = ['red', 'green', 'blue', 'yellow']
    for x, y in zip(x_transformed, y_label):
        ax.scatter(x[0], x[1], x[2], c=color[y])
    ax.set_title('pca practice in numpy')


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    pca_1(ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    pca_2(ax2)
    plt.show()

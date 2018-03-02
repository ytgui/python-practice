import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def load_datasets():
    x_data, y_label = datasets.make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=2.0)
    # x_data = (x_data - np.mean(x_data, axis=0)) / np.sqrt(np.var(x_data))

    n_samples, n_features = np.shape(x_data)
    idx = list(range(n_samples))
    random.shuffle(idx)
    x_data, y_label = x_data[idx], y_label[idx]

    return x_data, y_label


def lda_2():
    x_data, y_label = load_datasets()
    n_samples, n_features = np.shape(x_data)

    # positive and negative samples
    idx_positive = np.equal(y_label, 0)
    x_positive, x_negative = x_data[idx_positive], x_data[np.logical_not(idx_positive)]

    # means
    u_positive, u_negative = np.mean(x_positive, axis=0), np.mean(x_negative, axis=0)

    # center dist
    s_b = np.matmul(u_positive - u_negative, (u_positive - u_negative).T)

    # covariance matrix
    s_w = np.cov(x_positive - u_positive, rowvar=False) + np.cov(x_negative - u_negative, rowvar=False)

    #
    lamb, p = np.linalg.eigh(np.linalg.inv(s_w) * s_b)
    print(lamb)
    w = p[:, 0]
    k = - w[0] / w[1]
    print('w:', w, 'k:', k)

    # plot samples
    color = [('red', 'green')[label] for label in y_label]
    plt.scatter(x_data[:, 0], x_data[:, 1], c=color)

    plt.xlim(np.min(x_data[:, 0]) - 1, np.max(x_data[:, 0]) + 1)
    plt.ylim(np.min(x_data[:, 1]) - 1, np.max(x_data[:, 1]) + 1)

    # plot line
    X = np.linspace(np.min(x_data[:, 0]) - 1, np.max(x_data[:, 0])) + 1
    Y = k * X
    plt.plot(X, Y)

    # plot transformed samples
    for x1, y1 in x_data:
        # (x1, y1) is sample point
        # (x2, y2) is intersection
        # (y2 - y1) / (x2 - x1) = - 1 / k
        # y2 = k * x2
        x2 = (x1 + k * y1) / (k ** 2 + 1)
        y2 = k * x2
        plt.plot((x1, x2), (y1, y2), '--', linewidth=1)

    #
    plt.show()


if __name__ == '__main__':
    lda_2()

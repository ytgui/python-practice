import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def radius_neighbors(x_data, center, radius):
    dist = np.linalg.norm(x_data - center, axis=1)
    dist_order = np.argsort(dist)
    dist_order = dist_order[:np.searchsorted(dist[dist_order], radius)]
    neighbors = x_data[dist_order]
    return neighbors


def nearest_neighbors(x_data, centers):
    y_predict = np.zeros(shape=x_data.shape[0], dtype=np.int)
    for idx, x in enumerate(x_data):
        y_predict[idx] = np.argmin(np.linalg.norm(x - centers,axis=1))
    return y_predict


def estimate_bandwidth(x_data):
    # estimate a radius to choice nearest samples
    # scikit-learn implement this method really ugly
    # maybe there is not a common way to collect this param
    # so just return a pre-defined value
    return 2.5


def meanshift_single(mean, x_data, radius):
    for _ in range(100):
        neighbors = radius_neighbors(x_data, center=mean, radius=radius)
        mean_old = mean
        mean = np.mean(neighbors, axis=0)
        if np.linalg.norm(mean - mean_old) < 0.001:
            break
    return mean


def meanshift_2():
    x_data, y_label = datasets.make_blobs(n_samples=500, random_state=20)

    # estimate a radius
    bandwidth = estimate_bandwidth(x_data)

    # apply meanshift for x_data
    means = np.array([meanshift_single(x, x_data, bandwidth) for x in x_data])

    # remove duplicates
    means = np.round(means, decimals=1)
    means = np.unique(means, axis=0)

    # label
    y_predict = nearest_neighbors(x_data, means)

    # draw
    color = ['red', 'green', 'blue']
    plt.scatter(means[:, 0], means[:, 1], marker='x', linewidths=10)
    for x, y in zip(x_data, y_predict):
        plt.scatter(x[0], x[1], c=color[y])
    plt.show()


def main():
    meanshift_2()


if __name__ == '__main__':
    main()

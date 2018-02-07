import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cluster


n_samples = 1500
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=20)
# no_structure = np.random.rand(n_samples, 2), None


def kmeans_1():
    x_data, y_label = blobs
    y_predict = cluster.KMeans(n_clusters=3).fit_predict(x_data)

    color = ['red', 'green', 'blue']
    for x, y in zip(x_data, y_predict):
        plt.scatter(x[0], x[1], c=color[y])
    plt.show()


def main():
    kmeans_1()


if __name__ == '__main__':
    main()

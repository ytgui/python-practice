import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cluster


def init_center(n_clusters, x_data):
    # https://en.wikipedia.org/wiki/K-means++
    # empty array
    centers = list()

    # 1. Choose one center uniformly at random from among the data points.
    centers.append(x_data[np.random.randint(x_data.shape[0])])

    for i in range(1, n_clusters):
        # 2. For each data point x, compute D(x), the distance
        # between x and the nearest center that has already been chosen.
        dist = list(map(lambda x: np.min(np.linalg.norm(np.subtract(x, centers), axis=1)), x_data))

        # 3. Choose one new data point at random as a new center,
        # using a weighted probability distribution where a
        # point x is chosen with probability proportional to D(x)^2.
        idx = np.searchsorted(np.cumsum(dist), np.random.rand() * np.sum(dist))

        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        # we just get a new center by probability distribution, but scikit-learn
        # tries more times to generate a more stable result
        centers.append(x_data[idx])
    return centers


def assignment_step(x_data, centers):
    # Assign each observation to the cluster whose mean has the least squared Euclidean distance
    # Returns a label for each data in the dataset
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    y_predict = np.zeros(shape=(x_data.shape[0]), dtype=np.int)
    for i, x in enumerate(x_data):
        dist = np.linalg.norm(np.subtract(centers, x), axis=1)
        closest_idx = np.argmin(dist)
        y_predict[i] = closest_idx
    return y_predict


def update_step(n_clusters, x_data, y_label):
    # Calculate the new means to be the centroids of the observations in the new clusters.
    centers = np.zeros(shape=[n_clusters, 2])
    for i in range(n_clusters):
        cluster_idx = np.squeeze(np.equal(y_label, i))
        centers[i] = np.mean(x_data[cluster_idx], axis=0)
    return centers


def kmeans_2():
    # https://en.wikipedia.org/wiki/K-means_clustering
    # https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
    n_clusters = 3
    x_data, y_label = datasets.make_blobs(n_samples=300, random_state=20)
    centers = init_center(n_clusters, x_data)
    for i in range(300):
        y_predict = assignment_step(x_data, centers)
        new_centers = update_step(n_clusters, x_data, y_predict)
        loss = np.sum(np.linalg.norm(np.subtract(new_centers, centers), axis=1))
        if loss < 0.1:
            break
        centers = new_centers

        color = ['red', 'green', 'blue']
        for x, y in zip(x_data, y_predict):
            plt.scatter(x[0], x[1], c=color[y])
        plt.scatter(centers[:, 0], centers[:, 1], c='white', marker='x', linewidths=20)
        plt.draw()
        plt.pause(0.1)
    print('finish')
    plt.show()


def main():
    kmeans_2()


if __name__ == '__main__':
    main()

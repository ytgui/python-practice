import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cluster


def meanshift_1():
    x_data, y_label = datasets.make_blobs(n_samples=500, random_state=20)
    y_predict = cluster.MeanShift().fit_predict(x_data)

    color = ['red', 'green', 'blue']
    for x, y in zip(x_data, y_predict):
        plt.scatter(x[0], x[1], c=color[y])
    plt.show()


def main():
    meanshift_1()


if __name__ == '__main__':
    main()

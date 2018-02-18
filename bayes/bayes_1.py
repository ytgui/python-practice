import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, naive_bayes


def main_1():
    iris = datasets.load_iris()
    x_data, y_label = iris.data, iris.target
    idx = np.arange(0, x_data.shape[0], dtype=np.int)
    np.random.shuffle(idx)
    x_data, y_label = x_data[idx], y_label[idx]

    model = naive_bayes.GaussianNB()
    model.fit(x_data[:120], y_label[:120])
    y_predict = model.predict(x_data[120:])
    print(np.equal(y_predict, y_label[120:]))


def main_2():
    x_data, y_label = datasets.make_classification(n_samples=1000, weights=[0.2, 0.8])
    model = naive_bayes.GaussianNB()
    model.fit(x_data, y_label)
    print(model.class_prior_)


def main_3():
    x_data = datasets.load_boston().data
    naive_bayes.BernoulliNB()


if __name__ == '__main__':
    main_3()

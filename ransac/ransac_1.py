import random
import numpy as np
import matplotlib.pyplot as plt


def generate_datasets():
    n_samples = 100
    t = np.linspace(0, 5, n_samples)
    x = -1.2 * t * t + 1.8 * t + 0.5 + 1.5 * np.random.randn(n_samples)
    outer = [100 * random.random() if random.random() < 0.25 else 0.0 for _ in range(n_samples)]
    x -= outer
    return t, x


def fit_newton(x_data, y_target):
    # error = y_predict - y_target
    # loss = 0.5 * error * error
    # min loss => goal is to solve: loss' = e * e' = 0
    a, b, c = 1.0, 1.0, 0.0
    while True:
        g, h = np.zeros(shape=(1, 3)), np.zeros(shape=(3, 3))
        for x, y in zip(x_data, y_target):
            error = a * x ** 2 + b * x + c - y
            g += error * np.array([[x ** 2, x, 1], ])
            h += np.array([[x ** 4, x ** 3, x ** 2],
                           [x ** 3, x ** 2, x ** 1],
                           [x ** 2, x ** 1, x ** 0]])
        delta = np.matmul(g, np.linalg.inv(h))
        if np.sum(np.abs(delta)) < 1e-5:
            break
        a, b, c = a - delta[0, 0], b - delta[0, 1], c - delta[0, 2]
    return a, b, c


def ransac(x_data, y_target):
    n_samples, = x_data.shape
    best_param, best_loss = None, 999
    for _ in range(100):
        idx = list(range(n_samples))
        random.shuffle(idx)
        idx = idx[:int(0.25 * n_samples)]
        xx, yy = x_data[idx], y_target[idx]
        param = fit_newton(xx, yy)
        loss = np.sum(np.abs(yy - param[0] * xx ** 2 - param[1] * xx - param[2]))
        if loss < best_loss:
            best_loss = loss
            best_param = param
    return best_param


def main():
    t_data, x_data = generate_datasets()

    plt.subplot(211)
    plt.scatter(t_data, x_data, c='b', marker='.')
    param = fit_newton(t_data, x_data)
    plt.plot(t_data, param[0] * t_data * t_data + param[1] * t_data + param[2], c='r')

    plt.subplot(212)
    plt.scatter(t_data, x_data, c='b', marker='.')
    param = ransac(t_data, x_data)
    plt.plot(t_data, param[0] * t_data * t_data + param[1] * t_data + param[2], c='r')

    plt.show()


if __name__ == '__main__':
    main()

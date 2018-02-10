import numpy as np
import matplotlib.pyplot as plt


def generate_datasets():
    n_samples = 100
    t = np.linspace(0, 5, n_samples)
    x = -1.2 * t * t + 1.8 * t + 0.5 + 1.5 * np.random.randn(n_samples)
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


def main():
    t_data, x_data = generate_datasets()
    plt.scatter(t_data, x_data, c='b', marker='.')

    param = fit_newton(t_data, x_data)
    plt.plot(t_data, param[0] * t_data * t_data + param[1] * t_data + param[2], c='r')

    plt.show()


if __name__ == '__main__':
    main()

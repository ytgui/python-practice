import numpy as np
import matplotlib.pyplot as plt


def generate_datasets():
    n_samples = 100
    t = np.linspace(0, 5, n_samples)
    x = -1.2 * t * t + 1.8 * t + 0.5 + 1.5 * np.random.randn(n_samples)
    return t, x


def main():
    t_data, x_data = generate_datasets()
    plt.scatter(t_data, x_data, c='b', marker='.')

    param = np.polyfit(t_data, x_data, 2)
    plt.plot(t_data, param[0] * t_data * t_data + param[1] * t_data + param[2], c='r')

    plt.show()


if __name__ == '__main__':
    main()

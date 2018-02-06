import random
import numpy as np
import matplotlib.pyplot as plt


def generate_1():
    n_points = 50
    t = np.linspace(1e-2, 1, n_points)
    x1 = 10 * np.sin(2 * np.pi * (25 + 0.1 * np.random.randn(1)) * t) + 5 * np.random.randn(n_points)
    x2 = 10 * np.cos(2 * np.pi * (25 + 0.1 * np.random.randn(1)) * t) + 5 * np.random.randn(n_points)
    x3 = 20 * t - 10 + 5 * np.random.randn(n_points)
    x4 = -20 * t + 10 + 5 * np.random.randn(n_points)
    y = random.randrange(4)
    x = [x1, x2, x3, x4][y]
    return x, y


def main():
    x_data = []
    y_label = []
    for _ in range(300):
        x, y = generate_1()
        x_data.append(x)
        y_label.append(y)
    np.savetxt('x_data.csv', x_data)
    np.savetxt('y_label.csv', y_label, fmt='%d')


if __name__ == '__main__':
    main()

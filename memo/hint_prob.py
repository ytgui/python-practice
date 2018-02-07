import numpy as np


def hint_prob():
    a = [33, 66, 1]
    b = np.cumsum(a)
    c = np.searchsorted(b, np.random.rand() * np.sum(a))
    return c


def validation():
    a1, a2, a3 = 0, 0, 0
    n = 100000
    for _ in range(n):
        c = hint_prob()
        if c == 0:
            a1 += 1
        elif c == 1:
            a2 += 1
        else:
            a3 += 1
    print(a1 / n, a2 / n, a3 / n)


if __name__ == '__main__':
    validation()

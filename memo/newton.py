def sqrt(n):
    # f = n - x^2 = 0
    # f' = -2 * x
    x = 1.0  # init value
    while True:
        delta = (n - x * x) / (-2 * x)
        if -1e-5 < delta < 1e-5:
            break
        x -= delta
    return x


if __name__ == '__main__':
    a = sqrt(2), sqrt(3), sqrt(4)
    print(a)

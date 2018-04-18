import random
import numpy as np


# ----------------------------------------------
# 米勒-罗宾 素数测试
# ----------------------------------------------
def miller_rabin(n, k=10):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if not n & 1:
        return False

    def check(a, s, d, n):
        x = pow(a, d, n)
        if x == 1:
            return True
        for i in range(s - 1):
            if x == n - 1:
                return True
            x = pow(x, 2, n)
        return x == n - 1

    s = 0
    d = n - 1

    while d % 2 == 0:
        d >>= 1
        s += 1

    for i in range(k):
        a = random.randrange(2, n - 1)
        if not check(a, s, d, n):
            return False
    return True


def test_miller_rabin():
    n = 2
    print(n, 'is' if miller_rabin(n) else 'is not', 'prime')
    n = 21
    print(n, 'is' if miller_rabin(n) else 'is not', 'prime')
    n = 53
    print(n, 'is' if miller_rabin(n) else 'is not', 'prime')
    n = 997
    print(n, 'is' if miller_rabin(n) else 'is not', 'prime')
    n = 65498423347
    print(n, 'is' if miller_rabin(n) else 'is not', 'prime')


# ----------------------------------------------
# 素数生成
# ----------------------------------------------
def generate_prime(n):
    primes = np.ones(shape=n, dtype=np.bool)
    primes[0] = primes[1] = False
    for i in range(2, n // 2 + 1):
        if primes[i]:
            for j in range(i ** 2, n, i):
                primes[j] = False
    return np.argwhere(primes).flatten()


def test_generate_prime():
    for prime in generate_prime(2 ** 16):
        if not miller_rabin(int(prime), k=16):
            print(prime)
    print('all tests passed')


def generate_big_prime():
    for i in range(1_0000_0000_0001, 2_0000_0000_0000, 2):
        if miller_rabin(i, k=16):
            print(i)


if __name__ == '__main__':
    generate_big_prime()

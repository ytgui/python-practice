import time
import random


# ----------------------------------------------
# 线性同余 伪随机数生成算法
# ----------------------------------------------
def lcg_1(seed=1):
    a, b = 0x343FD, 0x269EC3
    modulus = 65536
    while True:
        seed = (a * seed + b) % modulus
        yield seed


def lcg_2(seed=1):
    a, b = 0x343FD, 0x269EC3
    modulus = 2 ** 24
    while True:
        seed = (a * seed + b) % modulus
        yield seed


def test_lcg():
    # test lcg_1()
    history = set()
    seed = random.randrange(65536)
    for idx, item in enumerate(lcg_1(seed)):
        if item in history:
            print('repeat', item, 'in idx', idx)
            print('min', min(history), 'max', max(history))
            break
        history.add(item)

    # test lcg_2()
    history = set()
    seed = random.randrange(65536)
    for idx, item in enumerate(lcg_2(seed)):
        if item in history:
            print('repeat', item, 'in idx', idx)
            print('min', min(history), 'max', max(history))
            break
        history.add(item)


def main():
    test_lcg()


if __name__ == '__main__':
    main()

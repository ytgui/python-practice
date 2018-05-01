import hashlib


def dh_1():
    # 离散对数问题
    # # # # # # # # # #
    # 给定素数p和正整数g，知道
    # A = g ^ x % p
    # 已知 A g p，求 x 是困难的
    # # # # # # # # # #
    # 公开商讨参数
    g = 10
    p = 97

    # 生成各自私钥
    x1_private = 6
    x2_private = 8

    # 生成公钥
    a1 = g ** x1_private % p
    a2 = g ** x2_private % p

    # 交换公钥用自己私钥处理，得到秘密 s
    s1 = a1 ** x2_private % p
    s2 = a2 ** x1_private % p

    # 验证
    assert s1 == s2


def mod_inverse(x, p):
    # 模逆运算
    # 若有 ax % p = 1，已知 x p，求 a
    # 等效为 1 / x
    # 要求 p 为质数：https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
    inv1, inv2 = 1, 0
    while p != 1 and p != 0:
        inv1, inv2 = inv2, inv1 - inv2 * (x // p)
        x, p = p, x % p
    if p == 0:
        return None
    return inv2


def extended_gcd(x, y):
    # 递归法
    if y == 0:
        return x, 1, 0
    else:
        d, a, b = extended_gcd(y, x % y)
    return d, b, a - b * (x // y)


def generate_address(x, y):
    assert isinstance(x, bytes)
    assert isinstance(y, bytes)

    pass


def main():
    print(mod_inverse(3, 11))
    print(mod_inverse(4, 12))
    print(mod_inverse(5, 13))

    print(extended_gcd(3, 11))
    print(extended_gcd(4, 12))
    print(extended_gcd(5, 13))

    print(generate_address(b'', b''))


if __name__ == '__main__':
    main()

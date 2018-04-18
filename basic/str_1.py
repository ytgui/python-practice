def str_match(a, b):
    a, b = list(a), list(b)
    if a == b:
        return 0
    if len(a) < len(b):
        return -1
    idx_a = 0
    while idx_a < len(a):
        for idx_b in range(len(b)):
            if (idx_a + idx_b) >= len(a):
                break
            if a[idx_a + idx_b] != b[idx_b]:
                break
        else:
            return idx_a
        idx_a += 1
    return -1


def test_str_match():
    print(str_match('12345', '34'))


def main():
    pass


if __name__ == '__main__':
    main()

import numpy as np


# ----------------------------------------------
# 杨氏查找表
# 二维数组中：左上 < 中，右下 > 中
# 求在此表中的查找方法
# ----------------------------------------------
def young_search(table, target):
    h, w = table.shape
    val_min, val_max = table[0, 0], table[h - 1, w - 1]
    if target < val_min or target > val_max:
        return -1, -1

    row, col = 0, w - 1
    while row < h and col < w:
        if target == table[row, col]:
            return row, col
        elif target > table[row, col]:
            row += 1
        else:
            col -= 1
    return -1, -1


def test_young_search():
    table = np.array([[1, 3, 9],
                      [2, 4, 11],
                      [5, 10, 15]])
    assert young_search(table, 3) == (0, 1)
    assert young_search(table, 4) == (1, 1)
    assert young_search(table, 15) == (2, 2)
    assert young_search(table, 6) == (-1, -1)
    assert young_search(table, 16) == (-1, -1)


def main():
    pass


if __name__ == '__main__':
    main()

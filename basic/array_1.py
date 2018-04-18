import numpy as np


# ----------------------------------------------
# 2-sum 问题
# 给出一组数 array 和一个目标 target
# 求由数组内不同的两个元素构成 target
# ----------------------------------------------
def two_sum_1(array, target):
    visited = {}
    for i, num in enumerate(array):
        if target - num in visited:
            return [visited[target - num], i]
        visited[num] = i


def two_sum_2(array, target):
    array = np.array(array)
    idx_sort = np.argsort(array)
    array = array[idx_sort]

    left, right = 0, len(array) - 1
    while left < right:
        current = array[left] + array[right]
        if current == target:
            return int(idx_sort[left]), int(idx_sort[right])
        elif current > target:
            right -= 1
        else:
            left += 1
    return -1, -1


def test_two_sum():
    assert two_sum_1([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_1([2, 7, 11, 15], 17) == [0, 3]
    assert two_sum_1([2, 7, 11, 15], 26) == [2, 3]
    assert two_sum_2([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum_2([2, 7, 11, 15], 17) == (0, 3)
    assert two_sum_2([2, 7, 11, 15], 26) == (2, 3)


# ----------------------------------------------
# 3-sum 问题
# 给出一组数 array 和一个目标 target
# 求由数组内不同的三个元素构成 target
# tow_sum_subarray 一次只能返回一个结果，有待改进
# ----------------------------------------------
def three_sum(array, target):
    array = sorted(array)

    def tow_sum_subarray(subarray, subtarget, left, right):
        while left < right:
            current = subarray[left] + subarray[right]
            if current == subtarget:
                return True, subarray[left], subarray[right]
            elif current > subtarget:
                right -= 1
            else:
                left += 1
        return False, None, None

    result = []
    for idx_1, item_1 in enumerate(array):
        ret, item_2, item_3 = tow_sum_subarray(array, target - item_1, idx_1 + 1, len(array) - 1)
        if ret:
            result.append([item_1, item_2, item_3])
    return result


def main():
    three_sum([-2, 0, 1, 1, 2], 0)


if __name__ == '__main__':
    main()

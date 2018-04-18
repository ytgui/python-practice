import time
import random
import numpy as np


def bubble_sort(x):
    length = len(x)
    for i in range(length):
        for j in range(length - 1 - i):
            if x[j] > x[j + 1]:
                x[j], x[j + 1] = x[j + 1], x[j]
    return x


def insertion_sort(x):
    length = len(x)
    for j in range(1, length):
        key = x[j]
        # find appropriate position i
        i = j - 1
        while i >= 0 and x[i] > key:
            x[i + 1] = x[i]  # move backward
            i -= 1
        # insert key into this position
        x[i + 1] = key
    return x


def merge_sort(x):
    # recursive return condition
    length = len(x)
    if length in (0, 1):
        return x

    # split
    middle = length // 2
    left = merge_sort(x[:middle])
    right = merge_sort(x[middle:])

    # merge
    def merge(a, b):
        c = []
        while len(a) != 0 and len(b) != 0:
            if a[0] < b[0]:
                c.append(a[0])
                a.remove(a[0])
            else:
                c.append(b[0])
                b.remove(b[0])
        # deal odd number
        if len(a) != 0:
            c += a
        else:
            c += b
        return c
    return merge(left, right)


def quick_sort(xx):
    def sort(x, left, right):
        # recursive return condition
        if left >= right:
            return

        # save marker
        low, high = left, right

        #
        key = x[low]
        while left < right:
            # left <- right, find the first item which smaller than key
            while left < right and x[right] > key:
                right -= 1
            x[left] = x[right]

            # left -> right, find the first item which bigger than key
            while left < right and x[left] <= key:
                left += 1
            x[right] = x[left]
        x[right] = key

        sort(x, low, left - 1)
        sort(x, left + 1, high)
    sort(xx, 0, len(xx) - 1)
    return xx


def main():
    # bubble sort
    y = bubble_sort([9, 4, 3, 8, 6, 0, 1, 2, 5, 7])
    print(y)

    # insertion sort
    y = insertion_sort([9, 4, 3, 8, 6, 0, 1, 2, 5, 7])
    print(y)

    # merge sort
    y = merge_sort([9, 4, 3, 8, 6, 0, 1, 2, 5, 7])
    print(y)

    # quick sort
    y = quick_sort([9, 4, 3, 8, 6, 0, 1, 2, 5, 7])
    print(y)


if __name__ == '__main__':
    main()

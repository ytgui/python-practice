import numpy as np


a = [[[1, 2],
      [3, 4]],
     [[10, 20],
      [30, 40]],
     [[100, 200],
      [300, 400]]]
print(np.shape(a))
print(np.sum(a, axis=0))


b = [[1, 2, 3],
     [4, 5, 6]]
print(np.shape(b))
print(np.sum(b, axis=0))

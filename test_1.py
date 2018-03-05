import numpy as np
import scipy


a = np.array([[1, 1],
              [0, 0],
              [-1, -1]])
b = np.linalg.norm(a, axis=1)
print(b)

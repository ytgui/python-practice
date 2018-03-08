import numpy as np
import scipy


a = np.array([1, 2, 3, 3, 5])
b = np.searchsorted(a, 2.1)
print(b)

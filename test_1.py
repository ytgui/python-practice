import numpy as np


a = [6, 1, 2, 3, 4, 5]
b = np.searchsorted(a, 2.1, sorter=True)
print(b)

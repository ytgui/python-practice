import numpy as np


a = [1, 2, 3]
b, c = np.average(a, returned=True)
print(b, c)
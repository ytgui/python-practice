import numpy as np


a = [1, 2, 2, 3]
b = np.unique(a)
print(a, b)


c = [1.2213, 10.134, 10.1123, 20.4575]
d = np.round(c, decimals=1)
e = np.unique(d)
print(c, d, e)

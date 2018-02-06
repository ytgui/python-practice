import numpy as np


a = [[1, 2],
     [3, 4]]

b = np.std(a, axis=0)
c = np.sqrt(np.var(a, axis=0))
print(np.allclose(b, c))

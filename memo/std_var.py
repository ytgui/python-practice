import numpy as np


a = [1, 3, 5]

b = np.std(a, axis=0)
c = np.sqrt(np.var(a, axis=0))
print(np.allclose(b, c))

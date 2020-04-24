import numpy as np

a = np.arange(1, 16, 1).reshape((3, 5))
print(a)
print(a[:, 1].reshape(1, -1))


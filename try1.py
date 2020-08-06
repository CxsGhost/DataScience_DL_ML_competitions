import numpy as np

a = np.ones(shape=(150, 1, 20))
b = np.ones(shape=(150, 20, 3))

c = np.matmul(a, b)

print(a.shape)
print(b.shape)
print(c.shape)

# 输出结果：
# (150, 1, 20)
# (150, 20, 3)
# (150, 1, 3)

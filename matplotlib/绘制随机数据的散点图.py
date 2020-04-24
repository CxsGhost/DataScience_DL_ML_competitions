import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(100)
y = np.random.random(100)
print(x)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=5, linewidths=1, c=[1, 0, 0], label="scatter")
plt.scatter(x, y, s=10, linewidths=2, marker='>', c=[0, 0, 1], label="scatter")
# c是颜色，一般要赋予数字，或者一个序列
# linewidths是笔触宽度
# maker是散点的形状，具体参数可以查阅百度
# s是点的大小，如果赋予一个sequence，则每个点的大小随机。
plt.xlabel("this is x")
plt.ylabel("this is y")
plt.ylim(-2.0, 2.0)
plt.xlim(-2.0, 2.0)

plt.legend()
plt.show()

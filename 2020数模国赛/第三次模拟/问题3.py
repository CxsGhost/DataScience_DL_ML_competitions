#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

N = 10
K = 0.5
x_jia = 1.14216397
y_jia = 0.034338
v = 340


def calculate_po(x_, y_):
    return -y_ / (N - x_)


def calculate_c1(n_, x_, po_, k_):
    return (po_ + np.sqrt(1 + np.square(po_))) / np.power(n_ - x_, -k_)


def yinzi(n, k, x, c1):
    yinzi = 0.5 * (-c1 * np.power(n - x, 1 - k) / (1 - k) +
                   1 / c1 * np.power(n - x, 1 + k) / (1 + k))
    return yinzi

po = calculate_po(x_jia, y_jia)
c1 = calculate_c1(N, x_jia, po, K)
print(c1)
c2 = y_jia - yinzi(N, K, x_jia, c1)
print('c2:{}万米'.format(c2))
print('时间t：{}秒'.format(c2 * 10000 / 340))


x1 = np.arange(0, 10, 0.1)

def curve(x):
    y = yinzi(N, K, x, c1) + c2
    return y

y_list = curve(x1)
x2 = [10 for _ in range(26)]
y2 = np.arange(0, y_list[-1] + 0.2, 0.2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 9))
plt.plot(x1, y_list, marker='o', color='blue', linewidth=2, label='导弹飞行轨迹')
plt.plot(x2, y2, marker='*', color='black', label='逃窜轨迹')
plt.xlabel('东向距离', fontdict={'size': 12})
plt.ylabel('北向距离', fontdict={'size': 12})
plt.title('追击路径')
plt.legend(fontsize='xx-large', loc='best')
plt.savefig('追击图.png')
plt.show()

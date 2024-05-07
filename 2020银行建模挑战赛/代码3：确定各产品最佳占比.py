import numpy as np

a1 = 0.984262864917498
a2 = 0.0157371350825009
in_out_rate = 0.15168754

b1_list = []
c1_list = []

target_list = []

r1 = 0.003
r2 = 0.021
p1 = 0.028
p2 = 0.04

print('搜索参数范围...')
for b1 in np.arange(0.1, 1.05, 0.05):
    for c1 in np.arange(0.1, 1.05, 0.05):
        if a1 * b1 + a2 * c1 < in_out_rate:
            continue
        else:
            b1_list.append(b1)
            c1_list.append(c1)
            target = a1 * b1 * r1 + a1 * (1 - b1) * r2 + a2 * c1 * p1 + a2 * (1 - c1) * p2
            target_list.append(target)

ind = np.argmax(target_list)
b1 = b1_list[ind]
b2 = 1 - b1
c1 = c1_list[ind]
c2 = 1 - c1
print(b1, b2, c1, c2)
print('最佳占比为：')
print('活期存款：{}'.format(a1 * b1))
print('定期存款：{}'.format(a1 * b2))
print('活期理财：{}'.format(a2 * c1))
print('定期理财：{}'.format(a2 * c2))


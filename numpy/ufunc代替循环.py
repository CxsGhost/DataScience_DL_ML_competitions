import numpy as np
from time import time


def func_x(a):
    y = a + 1
    return y


list_1 = np.arange(1, 100000, 2)

time_1 = time()

for i in list_1:
    print(func_x(i))

end_1 = time() - time_1
print(end_1)

time_2 = time()

func_np = np.frompyfunc(func_x, 1, 1)
# 其实相当于用此方法代替了循环，frompyfunc中
# 第一个参数是目标函数
# 第二个是传入函数的参数个数
# 第三个是函数的返回值个数，可以修改此数值强制拦截返回值
# 返回值将会是一个列表组成
print(func_np(list_1))
print(time() - time_2)
# 最终根据time模块的计算，使用numpy后时间大大减少

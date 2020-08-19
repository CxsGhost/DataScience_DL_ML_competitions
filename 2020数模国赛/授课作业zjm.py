import math
import numpy as np
import decimal



def calculate_1(start, end, p, days):
    rest_days = 0
    p_ = 0
    if start - 2 >= 1:
        rest_days += start - 2
    if end + 2 <= 7:
        rest_days += 6 - end
    if rest_days:
        for i in range(rest_days + 1):
            number = math.factorial(rest_days) / (math.factorial(i) * math.factorial(rest_days - i))
            p_ += math.pow(p, i + days) * math.pow(1 - p, 7 - i - days) * number
        return p_
    return math.pow(p, days) * math.pow(1 - p, 7 - days)



def calculate_2(p):
    P = 0
    for days in range(3, 8):
        for day in range(1, 9 - days):
            P += calculate_1(day, day + days - 1, p, days)
    return P - math.pow(p, 6) * math.pow(1 - p, 1)


print('概率为：{}'.format(calculate_2(0.5)))



def Machin():
    return 4 * (4 * np.arctan(1 / 5) - np.arctan(1 / 239))



def Monte_Carlo():
    inside = 0
    iteration = 10000000
    for _ in range(iteration):
        if np.sum(np.power(np.random.rand(1, 2), 2)) <= 1:
            inside += 1
    decimal.getcontext().prec = 20
    return decimal.Decimal(inside) / decimal.Decimal(iteration) * 4



def Leibniz():
    iteration = 1000000
    pi = 0
    for i in range(iteration):
        if i % 2:
            pi -= 1 / (2 * i + 1)
        else:
            pi += 1 / (2 * i + 1)
    return 4 * pi


machin = Machin()
monte = Monte_Carlo()
leib = Leibniz()


print('\n梅钦公式所得：{}\n误差为：{}\n'.format(machin, np.abs(np.pi - machin)))
print('莱布尼茨级数100万项所得：{}\n误差为：{}\n'.format(leib, np.abs(np.pi - leib)))
print('蒙特卡洛法1000万点所得：{}\n误差为：{}'.format(monte, np.abs(decimal.Decimal(np.pi) - monte)))


"""
概率为：0.3671875

梅钦公式所得：3.1415926535897936
误差为：4.440892098500626e-16

莱布尼茨级数100万项所得：3.1415916535897743
误差为：1.0000000187915248e-06

蒙特卡洛法1000万点所得：3.1409600000000000
误差为：0.00063265358979311599796
"""

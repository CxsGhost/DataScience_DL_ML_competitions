# import numpy as np
#
# matrix = np.zeros(shape=(4, 4))
# a = np.fill_diagonal(matrix, 4, wrap=True)
# print(matrix)
# print(a)
# print(*matrix.shape)


def test():
    x = 0
    loc = locals()
    print(locals())
    exec('x += 1')
    print('x=%d' % x)
    print(loc)
    print(locals())




test()
# 输出：
# {'x': 0}
# x=0
# {'x': 1, 'loc': {...}}

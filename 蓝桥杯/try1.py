import numpy as np
# f1 = 1
# f2 = 1
# _matrix = np.array([[1, 1], [1, 0]])
# f_matrix = np.array([[f1], [f2]])
# number = int(input()) - 2
# # if number == -1 or 0,这样写是错的
# # 因为这会将前后分成两个判断体，当第一个返回Flase时，第二个是0，也就是说还会是Flase，GG
# if number == -1 or number == 0:
#     print(1)
# else:
#     while not number == 1:
#         if number % 2 == 0:
#             _matrix = np.dot(_matrix, _matrix)
#             number = number / 2
#             print(_matrix)
#             print(f_matrix)
#             if f_matrix[0][0] <= 0:
#                 print(number, "偶数")
#                 raise ValueError
#
#         elif number % 2 != 0:
#             number = number - 1
#             f_matrix = np.dot(_matrix, f_matrix)
#             _matrix = np.dot(_matrix, _matrix)
#             number = number / 2
#             print(_matrix)
#             print(f_matrix)
#             if f_matrix[0][0] <= 0:
#                 print(number, '奇数')
#                 raise ValueError
#     f_matrix = np.dot(_matrix, f_matrix)
#     print(f_matrix)

# a = np.array([[1, 1], [1, 0]], dtype='')
# for i in range(200):
#     a = np.dot(a, a)
#     print(a)
#     for j in a:
#         for k in j:
#             if k <= 0:
#                 print(i)
#                 print(a.dtype)
#                 raise ValueError
import numpy as np
f0 = 0
f1 = 1

_matrix = np.array([[1, 1], [1, 0]])
f_matrix = np.array([[f1], [f0]])
number = - 1
while not number == 1:
    if number % 2 == 0:
        _matrix = np.dot(_matrix, _matrix)
        number = number / 2

    elif number % 2 != 0:
        number = number - 1
        f_matrix = np.dot(_matrix, f_matrix)
        _matrix = np.dot(_matrix, _matrix)
        number = number / 2
f_matrix = np.dot(_matrix, f_matrix)
print(f_matrix[0][0])


n = int(input("输入n:"))

def Fibonacci(_n):

    if _n == 1 or _n == 2:
        return 1
    else:
        return Fibonacci(_n - 1) + Fibonacci(_n - 2)


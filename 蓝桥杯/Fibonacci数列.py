# 方法1
import numpy as np
f1 = 1
f2 = 1
_matrix = np.array([[1, 1], [1, 0]])
f_matrix = np.array([[f1], [f2]])
number = int(input()) - 2
# if number == -1 or 0,这样写是错的
# 因为这会将前后分成两个判断体，当第一个返回Flase时，第二个是0，也就是说还会是Flase，GG
if number == -1 or number == 0:
    print(1)
else:
    while not number == 1:
        if number % 2 == 0:
            _matrix = np.dot(_matrix, _matrix)
            number = number / 2

        elif number % 2 != 0:
            number = number - 1
            f_matrix = np.dot(_matrix, f_matrix)
            _matrix = np.dot(_matrix, _matrix)
            number = number / 2
    print(_matrix)
    print("----------")
    print(f_matrix)
    print("---------")
    f_matrix = np.dot(_matrix, f_matrix)
    print(f_matrix)
    # print(f_matrix[0][0] % 10007)

# 方法2
f1 = 1
f2 = 1
matrix_1 = [[1, 1], [1, 0]]
matrix_2 = [[], []]
f_matrix = [[f1], [f2]]
f_matrix_c = [[], []]
number = int(input()) - 2
if number == -1 or number == 0:
    print(1)
else:
    while not number == 1:
        if number % 2 == 0:
            matrix_2[0].append(matrix_1[0][0]**2 + matrix_1[1][0] * matrix_1[0][1])
            matrix_2[0].append(matrix_1[0][1] * matrix_1[0][0] + matrix_1[1][1] * matrix_1[0][1])
            matrix_2[1].append(matrix_1[0][0] * matrix_1[1][0] + matrix_1[1][0] * matrix_1[1][1])
            matrix_2[1].append(matrix_1[0][1] * matrix_1[1][0] + matrix_1[1][1]**2)
            matrix_1 = matrix_2
            matrix_2 = [[], []]  # 这一句是重中之重，是大坑，不加上就全部GG，因为不加就相当于：1和2两个矩阵还是都指向了一个内存地址的数组，没意义，2矩阵没能发挥他真正的中间人作用
            number = number / 2

        elif number % 2 != 0:
            number = number - 1
            f_matrix_c[0].append(matrix_1[0][0] * f_matrix[0][0] + matrix_1[1][0] * f_matrix[1][0])
            f_matrix_c[1].append(matrix_1[0][1] * f_matrix[0][0] + matrix_1[1][1] * f_matrix[1][0])
            f_matrix = f_matrix_c
            f_matrix_c = [[], []]

            matrix_2[0].append(matrix_1[0][0] ** 2 + matrix_1[1][0] * matrix_1[0][1])
            matrix_2[0].append(matrix_1[0][1] * matrix_1[0][0] + matrix_1[1][1] * matrix_1[0][1])
            matrix_2[1].append(matrix_1[0][0] * matrix_1[1][0] + matrix_1[1][0] * matrix_1[1][1])
            matrix_2[1].append(matrix_1[0][1] * matrix_1[1][0] + matrix_1[1][1] ** 2)
            matrix_1 = matrix_2
            matrix_2 = [[], []]
            number = number / 2
    f_matrix_c[0].append(matrix_1[0][0] * f_matrix[0][0] + matrix_1[1][0] * f_matrix[1][0])
    print(f_matrix_c[0][0])
    # print(f_matrix_c[0][0] % 10007)


# 方法3
def Fibonacci():

    f1 = 1
    f2 = 1
    fibonacci = [f2, f1]
    n = int(input("输入n:"))
    if n == 1 or n == 2:
        return 1
    else:
        for i in range(n - 2):
            fibonacci[0], fibonacci[1] = fibonacci[0] + fibonacci[1], fibonacci[0]
        return fibonacci[0] % 10007

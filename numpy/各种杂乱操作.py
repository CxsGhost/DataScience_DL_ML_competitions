import numpy as np
a = np.array([[8, 5],
              [5, 2]])
b = [2, 5, 48, 9]
a_ = [['2'], ['3'], ['5'], ['7'], ['11']]
aa = np.asfarray(a_).reshape(1, -1)  # 可以把字符整数转化为真正的整数
print(aa)
print(a.dtype)
print(a.shape)
"_________________________"
a = a.astype(np.float32)  # 转换类型
print(a)
'______________________________'
a = a.flatten()  # 把高纬数组展平成一维
print(a)
'___________________________________'
arr10 = np.argmax(a[0])  # 找出当前维度最大数的索引
arr11 = np.argmax(a, axis=0)  # 找出每一列最大数的索引，1是每一行, -1是表示仅在最后一个维度上
arr12 = np.argmax(a)  # 此时找出整个数组最大数的索引，但返回的索引是将数组展平后的索引
print()
'_________________________________________'
a = np.transpose(a, axes='(index, index, index,....)')  # 矩阵转置
# axes参数意味着如何转置，各个位置上分别是转置前为各个维度的索引，也就是可以随意重新组合高维数据
print(a)
np.trace()  # 进行对角线求和，对于高维矩阵非常好用.如果a为二维，则返回沿对角线的总和。如果a的尺寸较大，则返回沿对角线的总和数组。
np.diagonal()  # 返回对角线，对于高维矩阵难以使用
'__________________________________________-'
a.shape = 4, -1  # 操作原数组
print(a)

c = a.reshape(2, 2)  # 建立副本
c[0, 0] = 52
print(a)
'_____________________________________'
arr = np.zeros((2, 3), dtype=np.float32)
arr1 = np.ones((1, 4))
# 建立只有0或1的数组，括号中的参数是维度

arr3 = np.ones_like(a)
arr4 = np.zeros_like(a)
# 建立和a维度相同的1或0矩阵
'__________________________________________-'
d = np.arange(1, 10, 2)  # 生成一维数组
print(d)
print(type(d))
'____________________________________________'
arr5 = np.logspace(2, 5, 4, base=2)
# 这里实际是以2的2次方为开头，2的5次方为结尾。
# 并且包括扩这两个值，建立一个等比数列，项的个数为4
arr6 = np.linspace(0, 10, num=1000, endpoint=False)  # 也是一维数组
# 这是建立等差数列，方法同上，有头有尾，1000个项,endpoint可以使得去尾。
'_____________________________________________________'
e = np.e
pi = np.pi
print(e, pi)  # 数学上的e和π
'_______________________________________'
zhishu = np.exp('数组或数')
zhishu_1 = np.exp2('')  # 指数函数2和e，参数可以是数字或者数组，会对每一个数进行运算
zhishu_ = np.power('底数', '指数')  # 此函数可以自定义任何底数的指数函数，指数部分可以是数组

duishu_1 = np.log('')
duishu_2 = np.log10('')
duiishu_3 = np.log2('')  # 分别以e,2，10为底数的对数函数，参数和指数函数一样

# 实际上numpy提供了任何可以想到的数学函数，使用时百度即可
'________________________________________________________________________'
arr7 = np.random.randint(low=0, high=10, size=(3, 5))
# 随机整数矩阵，去尾
np.random.seed(1)  # 设置随机种子，来控制接下来randint的选择
arr8 = np.random.randint(10)
'__________________________________________'
np.random.shuffle(a)
# 打乱数组，对于高维数组只会打乱第一维数组，并且此函数没有返回值，直接对原数组操作
'_____________________________________________'
color = ['green', 'red', 'blue']
choice = np.random.choice(color, size=(2, 2), p=[0.8, 0.19, 0.01])
# 可以自定义概率来进行抽样，p中的概率加起来必须是1，size是返回的数组规模
'________________________________________________________'
arr9 = np.random.normal(loc=3, scale=4, size=(2, 2))
# 从正态分布中随机抽取，loc是均值，也就μ，scale是标准差，也就是σ
'_______________________________________________________-'
arr13 = np.array([[0, 2, np.nan],
                [1, np.nan, -6],
                [np.nan, -2, 1]])
print(np.isnan(arr13))  # 将会true或Flase，维度同矩阵，只有nan是true
# array([[False, False, True],
#        [False, True, False],
#        [True, False, False]])
print(arr13 > 13)  # 也会返回true flase矩阵，任何判断符都可以
print(~(arr > 13))  # ~()表示布尔值取反
'______________________________________________'
arr = np.array([[0, 2, 3],
                [1, 0, 0],
                [-3, 0, 0]])
x_ind, y_ind = np.where(arr != 0)  # 此时只有一个参数
print(repr(x_ind))  # x indices of non-zero elements
print(repr(y_ind))  # y indices of non-zero elements
# 将会返回索引，xy坐标，
'__________________________________________________'
np_filter = np.array([[True, False], [False, True]])
positives = np.array([[1, 2], [3, 4]])
negatives = np.array([[-2, -5], [-1, -8]])
print(repr(np.where(np_filter, positives, negatives)))

np_filter = positives > 2
print(repr(np.where(np_filter, positives, negatives)))

np_filter = negatives > 0
print(repr(np.where(np_filter, positives, negatives)))
# true的将与positive矩阵位置一一对应的进行替换
# Flase的将与negative矩阵位置一一对应进行替换
'_________________________________________________________'
arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])
print(repr(arr > 0))
print(repr(np.any(arr > 0, axis=0)))  # 不加第二个参数则是对数组全体。0是每列，1是每行，-1最后一维度的每一维
print(repr(np.any(arr > 0, axis=1)))
print(repr(np.all(arr > 0, axis=1)))
# array([[False, False, False],
#        [ True,  True, False],
#        [ True,  True,  True]])
# array([ True,  True,  True])
# array([False,  True,  True])
# array([False, False,  True])
'____________________________________'
f = "helloyouhdfia"
g = np.fromstring(f, dtype=np.int8)
print(g)
'_______________________________________'
h = np.dtype({'names': ['name', 'age'], 'formats': ['S32', 'i']})  # 32字节的字符串
print(h)
# 此处h实际上是定义了一种结构数组的类型方式，就好像正则表达式中的，compile（）
k = np.array([("wlx", 25), ("lhy", 19)], dtype=h)
print(k)
print(k[0]["name"])
print(k[0][0])
'--------------------------'
zore_matrix = np.zeros(shape=(5, 5))
zore_matrix.fill_diagonal(x=5)

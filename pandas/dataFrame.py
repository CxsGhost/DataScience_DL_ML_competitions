import pandas as pd
import numpy as np


df1 = pd.DataFrame([[1, 2],
                   [3, 4]],
                   index=['w', 'l'],
                   columns=[2000, 1113])
print(df1)
# dataframe可以创建二维数据，但是必须要有行列标签，这其实是一个表格
# 如果不提供index和columns，则会自动按照数字索引
'----------------------------------------------'
dic1 = {'w': [20, 00], 'l': [11, 13]}
df2 = pd.DataFrame(dic1)
print(df2)
# 如果从字典创建，啧key会自动归结到columns
'---------------------------------------------'
df3 = pd.DataFrame([[1.2, 3], [4, 5]])
print(df3.dtypes)
# dataframe中以列为单位向上转换类型，这个例子中，第一列会全部默认为浮点类型
'----------------------------------------------'
ser = pd.Series([1, 2], name='ooo')
print(ser)
df4 = df3.append(ser, ignore_index=True)  # 此处是忽略行名称,包括原dataframe和新添加的series

df5 = df1.append(ser)
print(df4)
print(df5)
# series无论如何都是一维数据，可以向dataframe添加
# name参数相当于行名称，index还是自动归结为列名称
# 但是注意，手动设置过的columns或series的index将不会应用到新添加的行，（或者说添加总是不会合并格式）
# 而是会新开辟两行，没设置过的才能直接添加在尾部
'--------------------------------------------'
df6 = df5.drop(labels='w', axis=0)  # 删除本行，或者可以根据列标签
print(df6)
df7 = df5.drop(index='w')
print(df7)
df8 = df5.drop(columns=2000, index='w')  # 可以同时删除行列
print(df8)
'--------------------------------'

# 如何组合多个dataframe
df_1 = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]},
                    index=['r1', 'r2'])
df_2 = pd.DataFrame({'c1': [5, 6], 'c2': [7, 8]},
                    index=['r1', 'r2'])
concat1 = pd.concat([df_1, df_2], axis=1)  # 如果指定行，啧自动按照想同行标签合并，如无相同则新开辟
concat2 = pd.concat([df_1, df_2], axis=0)  # 合并后不会自动重新整理索引
reinde = df1.reset_index(drop=True, inplace=True)   # drop意思是以第一个为0排序，inplace,True是对原数组操作，不会创新新副本返回

print(concat1)
print(concat2)
'-------------------------------------------------'
mlb_df1 = pd.DataFrame({'name': ['john doe', 'al smith', 'sam black', 'john doe'],
                        'pos': ['1B', 'C', 'P', '2B'],
                        'year': [2000, 2004, 2008, 2003]})
mlb_df2 = pd.DataFrame({'name': ['john doe', 'al smith', 'jack lee'],
                        'year': [2000, 2004, 2012],
                        'rbi': [80, 100, 12]})

print('{}\n'.format(mlb_df1))
print('{}\n'.format(mlb_df2))

mlb_merged = pd.merge(mlb_df1, mlb_df2)  # 根据所有相同列标签，选出信息相同的条目，然后合并
print('{}\n'.format(mlb_merged))
# 在不使用任何关键字参数的情况下，pd.merge使用它们的所有公共列标签将两个DataFrame联接在一起。在代码示例中，mlb_df1和之间的共同标签mlb_df2是name和year。
#
# 包含与公共列标签完全相同的值的行将被合并。因为'john doe'对于一年2000是在这两个mlb_df1和mlb_df2，它的行被合并。但是，'john doe'由于year 2003仅在中mlb_df1，因此其行未合并。
#
# 该pd.merge函数接受许多关键字参数，但通常不需要它们来正确合并两个DataFrame。
'---------------------------------------------'


# 索引编制
df9 = pd.DataFrame({'c1': [1, 2, 7], 'c2': [3, 4,  6],
                   'c3': [5, 6, 9]}, index=['r1', 'r2',  'r3'])
series1 = df9['c1']  # 将会返回series
print(series1)
dt1 = df9[['c1']]  # 将会返回dataframe
print(dt1)
#  以上方式其中的参数均被视为列标签

dt2 = df9[0: 2]  # 这将默认是按行来切片，且具有去尾性
dt3 = df9['r1': 'r3']  # 这也是按行来，但不具有去尾性
dt4 = df9['r1']  # 这会报错，单个标签索引默认是列标签去找
print(dt2)
print(dt3)
print(dt4)
'---------------------------------------------'
df = pd.DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6],
                   'c3': [7, 8, 9]}, index=['r1', 'r2', 'r3'])

print('{}\n'.format(df))

print('{}\n'.format(df.iloc[1]))  # 将会把r2这一行的数据切成一个series返回

print('{}\n'.format(df.iloc[[0, 2], [0, 2]]))   # 将会返回dataframe

bool_list = [False, True, True]
print('{}\n'.format(df.iloc[bool_list]))  # 将会返回true行的dataframe

print(df.loc[['r1', 'r2']])  # 当截取两行及以上时，要两层方括号，返回dataframe，否则报错
print(df.loc['r1'])  # loc与上面iloc方法相似，但是iloc使用行索引数字,loc使用行标签
print(df.loc[bool_list])  # 同样也可以使用bool值来切片

print(df.loc[['r1'], ['c1']])  # 并且可以通过传入列标签进行特定数据的切片
print(df.loc['r1', 'c1'])  # 这是有区别的，这只是截取了一个数，是一维，而上面，则是截取了dataframe，是二维的

# 简单来说二者不同在于一个用索引，一个用标签，并且都首先以行为参数
# 并且逗号表示隔开，不同于range（0， 2）中的逗号，且不能用：来切片索引
'--------------------------------------------------------'
'下面是详细的dataframe的索引方法'
# df['a']  # 取a列
# df[['a', 'b']]  # 取a、b列
#
# # ix可以用数字索引，也可以用index和column索引
# df.ix[0]  # 取第0行
# df.ix[0:1]  # 取第0行
# df.ix['one':'two']  # 取one、two行
# df.ix[0:2, 0]  # 取第0、1行，第0列
# df.ix[0:1, 'a']  # 取第0行，a列
# df.ix[0:2, 'a':'c']  # 取第0、1行，abc列
# df.ix['one':'two', 'a':'c']  # 取one、two行，abc列
# df.ix[0:2, 0:1]  # 取第0、1行，第0列
# df.ix[0:2, 0:2]  # 取第0、1行，第0、1列
#
# # loc只能通过index和columns来取，不能用数字
# df.loc['one', 'a']  # one行，a列
# df.loc['one':'two', 'a']  # one到two行，a列
# df.loc['one':'two', 'a':'c']  # one到two行，a到c列
# df.loc['one':'two', ['a', 'c']]  # one到two行，ac列
#
# # iloc只能用数字索引，不能用索引名
# df.iloc[0:2]  # 前2行
# df.iloc[0]  # 第0行
# df.iloc[0:2, 0:2]  # 0、1行，0、1列
# df.iloc[[0, 2], [1, 2, 3]]  # 第0、2行，1、2、3列
#
# # iat取某个单值,只能数字索引
# df.iat[1, 1]  # 第1行，1列
# # at取某个单值,只能index和columns索引
# df.at['one', 'a']  # one行，a列

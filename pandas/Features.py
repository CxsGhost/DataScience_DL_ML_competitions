import pandas as pd

df = pd.DataFrame({
    'T1': [10, 15, 8],
    'T2': [25, 27, 25],
    'T3': [16, 15, 10]})

print('{}\n'.format(df))

print('{}\n'.format(df.sum()))

print('{}\n'.format(df.sum(axis=1)))

print('{}\n'.format(df.mean()))

print('{}\n'.format(df.mean(axis=1)))  # 可以是用参数控制对行列的计算
'-------------------------------------------------------'

print(df.multiply(0.1))  # 默认axis是对列操作
print(df.multiply([0.1, 0.2, 0.3]))  # 会将三列分别乘对应的数
print(df.multiply([0.1, 0.2, 0.3], axis=0))  # 这里与之前相反，0表示对行操作

df1 = df.multiply([0.1, 0.2, 0.3])
print(df1.sum())  # 通过multiply函数，可以对数据进行加权计算，加权计算默认列为单数
'-------------------------------------'
df['www'] = df['T1'] - df['T2']   #可以通过这种方式添加列，类似于字典的添加方式
print(df)

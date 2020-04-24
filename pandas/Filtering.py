import pandas as pd
import numpy as np

df = pd.DataFrame({
    'playerID': ['bettsmo01', 'canoro01', 'cruzne02', 'ortizda01', 'cruzne02'],
    'yearID': [2016, 2016, 2016, 2016, 2017],
    'teamID': ['BOS', 'SEA', 'SEA', 'BOS', 'SEA'],
    'HR': [31, 39, 43, 38, 39]})

print('{}\n'.format(df))

cruzne02 = df['playerID'] == 'cruzne02'
print('{}\n'.format(cruzne02))

hr40 = df['HR'] > 40
print('{}\n'.format(hr40))

notbos = df['teamID'] != 'BOS'
print('{}\n'.format(notbos))
# 以上判断操作将会返回一个series的bool值
'---------------------------------------'
df = pd.DataFrame({
    'playerID': ['bettsmo01', 'canoro01', 'cruzne02', 'ortizda01', 'cruzne02'],
    'yearID': [2016, 2016, 2016, 2016, 2017],
    'teamID': ['BOS', 'SEA', 'SEA', 'BOS', 'SEA'],
    'HR': [31, 39, 43, 38, 39]})

print('{}\n'.format(df))

str_f1 = df['playerID'].str.startswith('c')
print('{}\n'.format(str_f1))

str_f2 = df['teamID'].str.endswith('S')
print('{}\n'.format(str_f2))

str_f3 = ~df['playerID'].str.contains('o')
print('{}\n'.format(str_f3))

# 以上startswith，endswith， contains，与标准库中处理字符串的函数功能相同
# 同样是返回bool值序列
'-------------------------------------'
df = pd.DataFrame({
    'playerID': ['bettsmo01', 'canoro01', 'cruzne02', 'ortizda01', 'cruzne02'],
    'yearID': [2016, 2016, 2016, 2016, 2017],
    'teamID': ['BOS', 'SEA', 'SEA', 'BOS', 'SEA'],
    'HR': [31, 39, 43, 38, 39]})

print('{}\n'.format(df))

isin_f1 = df['playerID'].isin(['cruzne02',
                               'ortizda01'])
print('{}\n'.format(isin_f1))

isin_f2 = df['yearID'].isin([2015, 2017])
print('{}\n'.format(isin_f2))

# isin函数也是返回布尔值，可以提供多个参数，有一个满足就是true
'---------------------------------------------------'
# pandas的NaN = np.nan,可以用isna和notna函数来检测
df = pd.DataFrame({
    'playerID': ['bettsmo01', 'canoro01', 'doejo01'],
    'yearID': [2016, 2016, 2017],
    'teamID': ['BOS', 'SEA', np.nan],
    'HR': [31, 39, 99]})

print('{}\n'.format(df))

isna = df['teamID'].isna()
print('{}\n'.format(isna))

notna = df['teamID'].notna()
print('{}\n'.format(notna))
'----------------------------------'

# 以上全部都是如何得到过滤条件的方法
# 下面介绍如何取得过滤后的数据
df = df['此处是过滤条件']
df3 = df[df['teamID'].isna()]
print(df3)


import pandas as pd
import numpy as np


df1 = pd.read_csv('name', index_col=2, header=None)  # header是防止把第一行当作列属性，导致不能用
# index_col表适用哪一列来做行标签，也就是把那一列提到前面来
'----------------------------------------------------'
df = pd.read_excel('data.xlsx', sheet_name=1)
print('{}\n'.format(df))

print('MIL DataFrame:')
df = pd.read_excel('data.xlsx', sheet_name='MIL')
print('{}\n'.format(df))

# Sheets 0 and 1
df_dict = pd.read_excel('data.xlsx', sheet_name=[0, 1])
print('{}\n'.format(df_dict[1]))

# All Sheets
df_dict = pd.read_excel('data.xlsx', sheet_name=None)
print(df_dict.keys())

# sheet_name可以是索引，标签，以及列表，如果是none则读取所有，
# 返回的多个列表以字典的存储
'--------------------------------------------------'

df2 = pd.read_json('name', orient='index')
# 将JSON数据的每个外键都视为列标签，而将每个内键都视为行标签。
# 但是，当我们设置时orient='index'，外键被视为行标签，而内键被视为列标签。
'------------------------------------'
print('{}\n'.format(mlb_df))

# Index is kept when writing
mlb_df.to_csv('data.csv')
df = pd.read_csv('data.csv')
print('{}\n'.format(df))

# Index is not kept when writing
mlb_df.to_csv('data.csv', index=False)  # 这是把mlb_df写入data.csv文件
df = pd.read_csv('data.csv')
print('{}\n'.format(df))
# 当我们不使用任何关键字参数时，to_csv会将行标签写为CSV文件的第一列。
# 如果行标签有意义，这很好，但是如果它们只是整数，则我们并不希望它们在CSV文件中。
# 在这种情况下，我们设置index=False，以指定不将行标签写入CSV文件。
'------------------------------------------'
print('{}\n'.format(mlb_df1))
print('{}\n'.format(mlb_df2))

with pd.ExcelWriter('data.xlsx') as writer:  # 先把这个Excel打开成一个写入表格，返回一个对象
    mlb_df1.to_excel(writer, index=False, sheet_name='NYY')
    mlb_df2.to_excel(writer, index=False, sheet_name='BOS')

df_dict = pd.read_excel('data.xlsx', sheet_name=None)
print(df_dict.keys())
print('{}\n'.format(df_dict['BOS']))
# 基本to_excel功能只会将单个DataFrame写入电子表格。
# 但是，如果要在一个Excel工作簿中编写多个电子表格
# 则首先将Excel文件加载到中，pd.ExcelWriter然后使用ExcelWriter作为的第一个参数to_excel。
'----------------------------------------------'
df.to_json('data.json', orient='index')
# orient参数同样作用，外层为行标签，里层为列标签
'----------------------------------------'




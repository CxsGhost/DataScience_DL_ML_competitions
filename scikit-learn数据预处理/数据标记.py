# 下面以一个内置的肿瘤数据来说明

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

print(bc.data)  # 查看数据集
print(bc.data.shape)  # 查看数据的个数，以及维度

print(bc.target)  # 查看每个数据的标记，这里分为良性和恶性肿瘤，以0，1表示
print(bc.target.shape)  # 查看标记的个数

print(bc.target_names)  # 查看标记的名称

malignant = bc.data[bc.target == 0]  # 截取恶性肿瘤的数据
print(malignant)

benign = bc.data[bc.target == 1]  # 截取良性肿瘤的数据
print(benign)


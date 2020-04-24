import pandas as pd

data_all = pd.read_csv("MNIST_all/mnist_train.csv", header=None)
for i in range(12):
    data = data_all.iloc[i * 5000:i * 5000 + 4999]
    data.to_csv("CUT_MNIST/mnist_train_{}.csv".format(i), index=False)

train_data = pd.read_csv("CUT_MNIST/mnist_train_0.csv", header=None)
for i in range(1, 12):
    part_data = pd.read_csv("CUT_MNIST/mnist_train_{}.csv".format(i), header=None)
    train_data = pd.concat([train_data, part_data], axis=0)

# dataframe合并后，索引不会自动合并，需要重新标记
train_data = train_data.reset_index(drop=True)




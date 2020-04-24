from keras.datasets import mnist
from keras import models  # 模型容器
from keras import layers  # 网络中填充的层
from keras.utils import to_categorical  # 对标签要进行转化（one-hot)


# 获取内置的mnist数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 进行预处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 转化为one-hot表示法
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建模型, Dense是全连接层，选择激第一个参数是节点数，依次是激活函数和输入数据大小（只第一层需要输入大小）
network = models.Sequential()
# 设置输入大小意味着是首个隐藏层,单条数据是784，第二个数据是批量维度，不给定具体则可以是任意
network.add(layers.Dense(100, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

# 编译网络，选择优化器，损失函数（多元交叉熵），以及监控指标（这里只选了准确率）
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 训练网络, epoch遍历五遍数据，设置mini—batch大小， 并且每epoch会输出一次损失和准确率（基于训练数据）
network.fit(x=train_images, y=train_labels, epochs=5, batch_size=128)

# 测试集数据,查看损失和准确率
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)











































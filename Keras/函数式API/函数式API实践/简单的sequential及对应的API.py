from keras import models
from keras import layers
from keras import Input
from keras import activations
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics
from keras.utils import to_categorical
import numpy as np

# SEQUENTIAL模型
seq_network = models.Sequential()
seq_network.add(layers.Dense(32, activation=activations.relu,
                             input_shape=(64,)))
seq_network.add(layers.Dense(32, activation=activations.relu))
seq_network.add(layers.Dense(10, activation=activations.softmax))

# 对应的函数式API
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation=activations.relu)(input_tensor)
x = layers.Dense(32, activation=activations.relu,
                 kernel_regularizer=regularizers.l2(0.01))(x)
output_tensor = layers.Dense(10, activation=activations.softmax)(x)

api_model = models.Model(inputs=input_tensor, outputs=output_tensor)

print(api_model.summary())

api_model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

# 随便生成数据训练一下
inputs_data = np.random.randint(low=1, high=10, size=(50, 64))
labels = np.random.randint(low=0, high=10, size=(50, ))
labels = to_categorical(labels, num_classes=10)

api_model.fit(x=inputs_data, y=labels,
              epochs=2,
              batch_size=2)
score = api_model.evaluate(x=inputs_data, y=labels)
print(score)

#%%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
import matplotlib.pyplot as plt
#%%
data = pd.read_excel('数据.xlsx', index_col=None, header=None)
data = data.values

scaler = MinMaxScaler(feature_range=(0, 1))

#%%

all_x = []
all_y = []

for i in range(data.shape[1] // 3):
    if (i + 1) % 2:
        all_x.append(data[:, i * 3:(i + 1) * 3])
    else:
        all_y.append(data[:, i * 3:(i + 1) * 3])
all_x = np.array(all_x)
all_y = np.array(all_y)
all_x = all_x[:-1, :, :]
print(all_x.shape)
all_x[:, :, 0] = all_x[:, :, 0] / np.max(all_x[:, :, 0])
all_x[:, :, 1] = all_x[:, :, 1] / np.max(all_x[:, :, 1])
all_x[:, :, 2] = all_x[:, :, 2] / np.max(all_x[:, :, 2])

#%%

network = keras.models.Sequential()
network.add(keras.layers.LSTM(3, dropout=0.3, return_sequences=True))
network.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),
                loss=keras.losses.mean_squared_error,
                metrics=[keras.metrics.mean_squared_error])
history = network.fit(x=all_x, y=all_y,
                      batch_size=1,
                      epochs=5)


#%%

loss = history.history['loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, color='r', marker='o', linestyle='--', label='tra_Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('LOSS_VALUE')
plt.legend()
plt.show()

import tensorflow.keras as keras


max_feature = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_feature)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

network = keras.models.Sequential()

network.add(keras.layers.Embedding(max_feature, 128,
                                   input_length=max_len,
                                   name='embed'))
network.add(keras.layers.Conv1D(64, 7, activation=keras.activations.relu))
network.add(keras.layers.MaxPool1D(5))
network.add(keras.layers.Conv1D(32, 7, activation=keras.activations.relu))
network.add(keras.layers.GlobalMaxPool1D())
network.add(keras.layers.Dense(1))

print(network.summary())

network.compile(optimizer=keras.optimizers.RMSprop(),
                loss=keras.losses.binary_crossentropy,
                metrics=[keras.metrics.binary_accuracy])

# 绘制由层组成的模型图
keras.utils.plot_model(network, show_shapes=True, show_layer_names=True, to_file='model.png')

# 使用一个tensorboard回调函数来训练模型
callbacks_list = [keras.callbacks.TensorBoard(log_dir='tensorboard_log_nicai',  # 日志文件被写入位置
                                              histogram_freq=1,  # 每epoch后记录激活直方图
                                              embeddings_freq=1)]  # 每epoch后记录嵌入数据

network.fit(x=x_train, y=y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks_list)





























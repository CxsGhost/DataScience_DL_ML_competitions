"""
详见212页
一下这个回调函数，在每epoch结束后将模型的每层的激活保存到指定位置
这个激活来自验证集第一个样本计算得到
"""
import numpy as np
import keras


# 继承callback类（其中两个函数都是父类中已有的，我们只需要重写成自己需要的即可）
class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):

        # 在训练之前由父模型调用，告诉回调函数哪个模型在调用它
        self.model = model

        # 取出该模型所有层的输出
        layer_outputs = [layer.output for layer in model.layers]

        # 真正的模型实例，返回每层的激活
        self.activations_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Require validation_data')

        # 取出第一个验证数据，放入模型中计算
        validation_sample = self.validation_data[0][0: 1]
        activations = self.activations_model.predict(validation_sample)
        file_ = open('activation_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(file_, activations)
        file_.close()


























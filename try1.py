
from tensorflow import keras
import kerastuner
# base_model = keras.applications.vgg16.VGG16(input_shape=(300, 300, 1),
#                                             weights=None,
#                                             include_top=True,
#                                             classes=10)
#
# print(base_model.summary())

print(dir(keras.backend))
keras.layers.Concatenate()
keras.layers.Average()

def build_model():
    model_input = keras.Input(shape=(20, 20),
                              dtype='float32')
    layer_1 = keras.layers.Dense(units=20,
                                 kernel_regularizer=keras.regularizers.L2(l2=0.1))(model_input)
    layer_2 = keras.layers.LeakyReLU()(layer_1)
    model_output = keras.layers.Dense(units=1, activation=keras.activations.sigmoid)(layer_2)
    model = keras.models.Model(inputs=[model_input], outputs=[model_output], name='ceshi')
    print(model.summary())
    return model

model = build_model()
model.compile(optimizer='rmsprop',
              loss=keras.losses.mae)

from kerastuner import HyperParameters

hp = HyperParameters()



























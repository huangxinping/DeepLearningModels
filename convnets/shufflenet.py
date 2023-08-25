from pyparsing import IndentedBlock
import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

print(f'Tensorflow version: {tf.__version__}')

class ChannelShuffle(tf.keras.layers.Layer):
    
    def __init__(self):
        super(ChannelShuffle, self).__init__()

    def call(self, inputs):
        # inputs shape is [batch, height, width, channels]
        # channels_per_group is 4
        channels_per_group = inputs.shape[-1] // 4
        x = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2], 4, channels_per_group]) # output: [batch, height, width, channels_per_group, channels/channels_per_group]
        x = tf.transpose(x, [0, 1, 2, 4, 3]) # 3和4维度调换一下。output: [batch, height, width, channels/channels_per_group, channels_per_group]
        x = tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[-1]])
        return x

def ShuffleNetUnitModule(inputs, in_channels, out_channels, strides):
    x = tf.keras.layers.Conv2D(filters=in_channels, kernel_size=(1, 1), groups=8, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = ChannelShuffle()(x)
    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Conv2D(filters=out_channels if strides < 2 else out_channels - in_channels, kernel_size=(1, 1), groups=8, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides < 2:
        x = tf.keras.layers.Add()([inputs, x])
    else:
        shortcut = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(inputs)
        x = tf.keras.layers.Concatenate()([shortcut, x])
    output = tf.keras.layers.Activation('relu')(x)
    return output

def StageModule(inputs, in_channels, out_channels, repeat):
    x = ShuffleNetUnitModule(inputs, in_channels, out_channels, strides=2)
    for _ in range(repeat):
        x = ShuffleNetUnitModule(x, out_channels, out_channels, strides=1)
    return x

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

# g = 1
x = StageModule(x, 24, 144, 3)
x = StageModule(x, 144, 288, 7)
x = StageModule(x, 288, 576, 3)
# # g = 2
# x = StageModule(x, 24, 200, 3)
# x = StageModule(x, 200, 400, 7)
# x = StageModule(x, 400, 800, 3)
# # g = 3
# x = StageModule(x, 24, 240, 3)
# x = StageModule(x, 240, 480, 7)
# x = StageModule(x, 480, 960, 3)
# # # g = 4
# x = StageModule(x, 24, 272, 3)
# x = StageModule(x, 272, 544, 7)
# x = StageModule(x, 544, 1088, 3)
# # # g = 8
# x = StageModule(x, 24, 384, 3)
# x = StageModule(x, 384, 768, 7)
# x = StageModule(x, 768, 1536, 3)

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='ShuffleNet')


if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'shufflenet_model.png'), show_shapes=True)
    # # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'shufflenet_layers.png')).show()
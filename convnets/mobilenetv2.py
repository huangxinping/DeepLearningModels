import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os


def InvertedResidualsModule(inputs, in_channels, out_channels, t, strides):
    x = tf.keras.layers.Conv2D(filters=in_channels * t, kernel_size=(1, 1), strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1, padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # if the strides are not 1, then we need to add a skip connection
    if strides == 1 and in_channels == out_channels:
        x =  tf.keras.layers.Add()([x, inputs])
    return x


input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = InvertedResidualsModule(x, in_channels=32, out_channels=16, t=1, strides=1)

x = InvertedResidualsModule(x, in_channels=16, out_channels=24, t=6, strides=2)
x = InvertedResidualsModule(x, in_channels=16, out_channels=24, t=6, strides=1)

x = InvertedResidualsModule(x, in_channels=24, out_channels=32, t=6, strides=2)
x = InvertedResidualsModule(x, in_channels=24, out_channels=32, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=24, out_channels=32, t=6, strides=1)

x = InvertedResidualsModule(x, in_channels=32, out_channels=64, t=6, strides=2)
x = InvertedResidualsModule(x, in_channels=32, out_channels=64, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=32, out_channels=64, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=32, out_channels=64, t=6, strides=1)

x = InvertedResidualsModule(x, in_channels=64, out_channels=96, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=64, out_channels=96, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=64, out_channels=96, t=6, strides=1)

x = InvertedResidualsModule(x, in_channels=96, out_channels=160, t=6, strides=2)
x = InvertedResidualsModule(x, in_channels=96, out_channels=160, t=6, strides=1)
x = InvertedResidualsModule(x, in_channels=96, out_channels=160, t=6, strides=1)

x = InvertedResidualsModule(x, in_channels=160, out_channels=320, t=6, strides=1)

x = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=1, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='MobileNet-V2')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'mobilenetv2_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'mobilenetv2_layers.png')).show()

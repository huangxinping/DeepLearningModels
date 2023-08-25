import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os
import math


def MBConvModule(inputs, filters_in, filters_out, kernel_size=3, strides=1, exp_ratio=6, se_ratio=0.25):
    """
    Arguments:
    -------
    input: input tensor
    filters_in: input filters
    filters_out: output filters
    kernel_size: the size/dimension of convolution filters
    strides: integer, the stride of convolution. If strides=2, padding in depthwise conv is 'valid'.
    exp_ratio: expansion ration, an integer for scaling the input filters/filters_in
    se_ratio: a float between 0 and 1 for squeezing the input filters
    -------
    """

    # Expansion layer: exp_ratio is integer >=1
    
    filters = filters_in * exp_ratio
    if exp_ratio != 1:
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.swish(x)
    else:
        x = inputs

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    # Squeeze and excitation
    if se_ratio > 0 and se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio)) #max with 1 to make sure filters are not less than 1
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        se = tf.keras.layers.Conv2D(filters_se, kernel_size=1, padding='same', activation='swish')(se)
        se = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation='sigmoid')(se)
        x = tf.keras.layers.multiply([x, se])

    x = tf.keras.layers.Conv2D(filters_out, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add identity shortcut if strides=2 and in & filters are same
    if strides == 1 and filters_in == filters_out:
        x = tf.keras.layers.add([x, inputs])
    return x


# Setting some hyperparameters for EfficientNet-B0
input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.swish(x)

# MBConv blocks
# Block 1: input filters=32, output filters=16, kernel size=3, block repeats=1
x = MBConvModule(x, filters_in=32, filters_out=16, kernel_size=3, strides=1, exp_ratio=1)

# Block 2: input filters=16, output filters=24, kernel size=3, strides=2, block repeats=2
# the first block of every stage has stride of 1
x = MBConvModule(x, filters_in=16, filters_out=24, kernel_size=3, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=16, filters_out=24, kernel_size=3, strides=2, exp_ratio=6)

# Block 3: input filters=24, output filters=40, kernel size=5, strides=2, block repeats=2 
x = MBConvModule(x, filters_in=24, filters_out=40, kernel_size=5, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=24, filters_out=40, kernel_size=5, strides=2, exp_ratio=6)

# Block 4: input filters=40, output filters=80, kernel size=3, strides=2, block repeats=3
x = MBConvModule(x, filters_in=40, filters_out=80, kernel_size=3, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)
x = MBConvModule(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)
x = MBConvModule(x, filters_in=40, filters_out=80, kernel_size=3, strides=2, exp_ratio=6)

# Block 5: input filters=80, output filters=112, kernel size=5, strides=1, block repeats=3
x = MBConvModule(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=80, filters_out=112, kernel_size=5, strides=1, exp_ratio=6)

# Block 6: input filters=112, output filters=192, kernel size=5, strides=2, block repeats=4
x = MBConvModule(x, filters_in=112, filters_out=192, kernel_size=5, strides=1, exp_ratio=6)
x = MBConvModule(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)
x = MBConvModule(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)
x = MBConvModule(x, filters_in=112, filters_out=192, kernel_size=5, strides=2, exp_ratio=6)

# Block 7: input filters=192, output filters=320, kernel size=3, strides = 1, block repeats=1
x = MBConvModule(x, filters_in=192, filters_out=320, kernel_size=3, strides=1, exp_ratio=6)

# Classification head
x = tf.keras.layers.Conv2D(filters=1280, kernel_size=1, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.swish(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='EfficientNet-B0')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'efficientnetb0_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'efficientnetb0_layers.png')).show()

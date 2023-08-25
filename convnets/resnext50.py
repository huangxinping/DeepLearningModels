import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

print(f'Tensorflow version: {tf.__version__}')

#  
# Unfortunately, the model is not working on CPUs.
#
# Grouped convolutions are currently only supported on GPUs, not CPUs.
# For now grouped convolutions on CPU are only supported using XLA.
#
# References: https://github.com/tensorflow/tensorflow/issues/29005
#             https://github.com/tensorflow/tensorflow/issues/34024
#             https://discuss.tensorflow.org/t/error-message-for-grouped-convolution-backprop-on-cpu-is-uninformative/6323
#

def IdentityModule(inputs, dim):
    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=(1, 1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=(3, 3), groups=32, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
        
    x = tf.keras.layers.Conv2D(filters=2*dim, kernel_size=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([inputs, x])
    output = tf.keras.layers.Activation('relu')(x)
    return output

def ProjectionModule(inputs, dim, strides=1):
    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=(1, 1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=(3, 3), strides=strides, groups=32, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
        
    x = tf.keras.layers.Conv2D(filters=2*dim, kernel_size=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = tf.keras.layers.Conv2D(filters=2*dim, kernel_size=(1, 1), strides=strides)(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    output = tf.keras.layers.Activation('relu')(x)
    return output

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = ProjectionModule(x, 128) # x3
for _ in range(2):
    x =IdentityModule(x, 128) 

x = ProjectionModule(x, 256, 2) # x4
for _ in range(3):
    x = IdentityModule(x, 256)
    
x = ProjectionModule(x, 512, 2) # x6
for _ in range(4):
    x = IdentityModule(x, 512)   
    
x = ProjectionModule(x, 1024, 2) # x3
for _ in range(2):
    x = IdentityModule(x, 1024)    

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='ResNeXt-50')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'resnext_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'resnext_layers.png')).show()

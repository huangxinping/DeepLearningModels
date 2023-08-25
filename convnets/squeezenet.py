import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

    
def FireModule(inputs, c1, c2):
    # squeeze
    x = tf.keras.layers.Conv2D(filters=c1, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    # expand
    x11 = tf.keras.layers.Conv2D(filters=c2, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x12 = tf.keras.layers.Conv2D(filters=c2, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # output
    x = tf.keras.layers.Concatenate()([x11, x12])
    return x
        
        
input = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

x = FireModule(x, 16, 64) # fire2
x = FireModule(x, 16, 64) # fire3
x = FireModule(x, 32, 128) # fire4
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)
x = FireModule(x, 32, 128) # fire5
x = FireModule(x, 48, 192) # fire6
x = FireModule(x, 48, 192) # fire7
x = FireModule(x, 64, 256) # fire8
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)
x = FireModule(x, 64, 256) # fire9
x = tf.keras.layers.Dropout(0.5)(x)

# Units in last layer are 3 per rps dataset
x = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.GlobalAvgPool2D()(x)
output = tf.keras.layers.Activation('relu')(x)
model = tf.keras.Model(inputs=input, outputs=output, name='SqueezeNet')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'squeezenet_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'squeezenet_layers.png')).show()

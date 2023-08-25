import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os


def IdentityModule(inputs, filters, strides=(1, 1)):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[0], padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[1], padding='same')(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ProjectionModule(inputs, filters, strides=(1, 1, 2)):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[0], padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[1], padding='same')(x)
    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides[2], padding='same')(inputs)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = IdentityModule(x, filters=64)
x = IdentityModule(x, filters=64)

x = ProjectionModule(x, filters=128, strides=(2, 1, 2))
x = IdentityModule(x, filters=128)

x = ProjectionModule(x, filters=256, strides=(2, 1, 2))
x = IdentityModule(x, filters=256)

x = ProjectionModule(x, filters=512, strides=(2, 1, 2))
x = IdentityModule(x, filters=512)

x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='ResNet18')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'resnet18_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'resnet18_layers.png')).show()

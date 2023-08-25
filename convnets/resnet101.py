import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os


def IdentityModule(inputs, filters):
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ProjectionModule(inputs, filters):
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=2)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=2)(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = ProjectionModule(x, filters=(64, 64, 256)) # 1
x = IdentityModule(x, filters=(64, 64, 256))
x = IdentityModule(x, filters=(64, 64, 256)) # 3

x = ProjectionModule(x, filters=(128, 128, 512)) # 1
x = IdentityModule(x, filters=(128, 128, 512))
x = IdentityModule(x, filters=(128, 128, 512))
x = IdentityModule(x, filters=(128, 128, 512)) # 4

x = ProjectionModule(x, filters=(256, 256, 1024)) # 1
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) # 5
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) # 10
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) # 15
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) # 20
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) 
x = IdentityModule(x, filters=(256, 256, 1024)) # 23

x = ProjectionModule(x, filters=(512, 512, 2048)) # 1
x = IdentityModule(x, filters=(512, 512, 2048))
x = IdentityModule(x, filters=(512, 512, 2048)) # 3

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='ResNet101')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'resnet101_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'resnet101_layers.png')).show()

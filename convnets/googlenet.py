import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

    
def InceptionV1Module(inputs, c1, c2, c3, c4):
    x11 = tf.keras.layers.Conv2D(filters=c1, kernel_size=(1, 1), activation='relu')(inputs)
    
    x21 = tf.keras.layers.Conv2D(filters=c2[0], kernel_size=(1, 1), activation='relu')(inputs)
    x22 = tf.keras.layers.Conv2D(filters=c2[1], kernel_size=(3, 3), padding='same', activation='relu')(x21)
    
    x31 = tf.keras.layers.Conv2D(filters=c3[0], kernel_size=(1, 1), activation='relu')(inputs)
    x32 = tf.keras.layers.Conv2D(filters=c3[1], kernel_size=(5, 5), padding='same', activation='relu')(x31)

    x41 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(inputs)   
    x42 = tf.keras.layers.Conv2D(filters=c4, kernel_size=(1, 1), activation='relu')(x41)
    
    output = tf.keras.layers.Concatenate()([x11, x22, x32, x42])
    return output
        
        
input = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = InceptionV1Module(x, c1=64, c2=(96, 128), c3=(16, 32), c4=32)
x = InceptionV1Module(x, c1=128, c2=(128, 192), c3=(32, 96), c4=64)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = InceptionV1Module(x, c1=192, c2=(96, 208), c3=(16, 48), c4=64)
x = InceptionV1Module(x, c1=160, c2=(112, 224), c3=(24, 64), c4=64)
x = InceptionV1Module(x, c1=128, c2=(128, 256), c3=(24, 64), c4=64)
x = InceptionV1Module(x, c1=112, c2=(144, 288), c3=(32, 64), c4=64)
x = InceptionV1Module(x, c1=256, c2=(160, 320), c3=(32, 128), c4=128)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = InceptionV1Module(x, c1=256, c2=(160, 320), c3=(32, 128), c4=128)
x = InceptionV1Module(x, c1=384, c2=(192, 384), c3=(48, 128), c4=128)
x = tf.keras.layers.GlobalAvgPool2D()(x)

x = tf.keras.layers.Dropout(0.4)(x)

# Units in last layer are 3 per rp dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='GoolgeNet-InceptionV1')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'googlenet_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'googlenet_layers.png')).show()

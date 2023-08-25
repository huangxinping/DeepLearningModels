import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)), # 1
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 2
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 3
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 4
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 5
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 6
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 7
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 8
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 9
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 10
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 11
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 12
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'), # 13
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(units=4096, activation='relu'), # 14
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(units=4096, activation='relu'), # 15
    tf.keras.layers.Dropout(0.5),

    # Units in last layer are 3 per rps dataset
    tf.keras.layers.Dense(units=3, activation='softmax') # 16
], name='VGG16')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'vgg16_model.png'), show_shapes=True)
    keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'vgg16_layers.png')).show()

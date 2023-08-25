import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
    
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),    
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
    
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(units=4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Units in last layer are 3 per rps dataset
    tf.keras.layers.Dense(units=3, activation='softmax')
], name='ZFNet')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'zfnet_model.png'), show_shapes=True)
    keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'zfnet_layers.png')).show()

import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation=tf.nn.sigmoid, input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='same', activation=tf.nn.sigmoid),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), strides=1, padding='same', activation=tf.nn.sigmoid),   
     
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(units=84, activation=tf.nn.sigmoid),

    # Units in last layer are 3 per rps dataset
    tf.keras.layers.Dense(units=3, activation='softmax')
], name='LeNet5')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'lenet5_model.png'), show_shapes=True)
    keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'lenet5_layers.png')).show()

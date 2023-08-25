import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

print(f'Tensorflow version: {tf.__version__}')


model = tf.keras.Sequential([
    # 1 layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 2 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 3 layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 4  layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 5 layer
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 6 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 7 layer
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 8 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 9 layer
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 10 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 10 layer
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 12 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 13 layer
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 14.1 layer (repeated 5 times)
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 14.2 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 14.3 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 14.4 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 14.5 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 15
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 16 layer
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 17 layer
    tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 18 layer
    tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # 19 layer
    tf.keras.layers.GlobalAveragePooling2D(),
    # 20 layer
    # Units in last layer are 3 per rps dataset
    tf.keras.layers.Dense(units=3, activation='softmax')
], name='MobileNet-V1')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'mobilenetv1_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'mobilenetv1_layers.png')).show()

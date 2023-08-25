import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

def convmixer_block(input, filters, kernel_size):
  """
  Input params
  ------
  input: input tensor
  filters: the number of output channels or filters in pointwise convolution
  kernel_size: kernel_size in depthwise convolution
  """
  shortcut = input

  # Depthwise convolution
  x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')(input)
  x = tf.keras.activations.gelu(x)
  x = tf.keras.layers.BatchNormalization()(x)

  # Shortcut connection
  x = tf.keras.layers.Add()([shortcut, x])

  # Pointwise or 1x1 convolution
  x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same')(x)
  x = tf.keras.activations.gelu(x)
  x = tf.keras.layers.BatchNormalization()(x)

  return x


input = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(filters=1536, kernel_size=7, strides=7)(input)
x = tf.keras.activations.gelu(x)
x = tf.keras.layers.BatchNormalization()(x)

# ConvMixer blocks repeated depth times
for _ in range(20):
  x = convmixer_block(x, 1536, 9)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='ConvMixer')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'convmixer_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'convmixer_layers.png')).show()

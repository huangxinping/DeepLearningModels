import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os


def RegMobule(input, filters_out, group_width=48, strides=1):
  """
  RegNetX-032 block (RegNetX blocks don't have squeeze and excitation)

  input: input tensor
  filters_out: output filters
  group_width: group widths in grouped convolutions
  strides: strides in grouped convolution. 
  
  If strides=1, we add identity shortcut from input to output. 
  If strides=2, the shortcut has 1x1 conv with strides 2.
  """

  groups = filters_out // group_width

  x = tf.keras.layers.Conv2D(filters=filters_out, kernel_size=1, strides=1)(input)
  x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(filters=filters_out, kernel_size=3, strides=strides, groups=groups, padding='same')(x)
  x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(filters=filters_out, kernel_size=1, strides=1)(x)
  x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
  x = tf.keras.layers.ReLU()(x)

  if strides == 1:
    shortcut = input
    x = tf.keras.layers.Add()([x, shortcut])
  elif strides == 2:
    shortcut = tf.keras.layers.Conv2D(filters=filters_out, kernel_size=1, strides=strides)(input)
    shortcut = tf.keras.layers.BatchNormalization(epsilon=1e-5)(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
  return x

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
x = tf.keras.layers.ReLU()(x)

# stage 1: 2 blocks, 96 channels(filters_out)
x = RegMobule(x, 96, group_width=48, strides=2)
x = RegMobule(x, 96, group_width=48, strides=1)

# stage 2: 6 blocks, 192 channels(filters_out)
x = RegMobule(x, 192, group_width=48, strides=2)
for _ in range(1, 6):
  x = RegMobule(x, 192, group_width=48, strides=1)

# stage 3: 15 blocks, 432 channels(filters_out)
x = RegMobule(x, 432, group_width=48, strides=2)
for _ in range(1, 15):
  x = RegMobule(x, 432, group_width=48, strides=1)

#stage 4: 2 blocks, 1008 channels(filters_out)
x = RegMobule(x, 1008, group_width=48, strides=2)
x = RegMobule(x, 1008, group_width=48, strides=1)

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='RegNetX-032')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'regnet_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'regnet_layers.png')).show()

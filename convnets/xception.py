import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

print(f'Tensorflow version: {tf.__version__}')


def EntryFlowModule(inputs):
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    shortcut = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding='same')(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    x = tf.keras.layers.Add()([shortcut, x])
    
    shortcut = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding='same')(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    x = tf.keras.layers.Add()([shortcut, x])
    
    shortcut = tf.keras.layers.Conv2D(filters=728, kernel_size=(1, 1), strides=2, padding='same')(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    x = tf.keras.layers.Add()([shortcut, x])
    return x

def MiddleFlowModule(inputs):
    x = tf.keras.layers.ReLU()(inputs)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([inputs, x])    
    return x

def ExitFlowModule(inputs):
    shortcut = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=2, padding='same')(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.ReLU()(inputs)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(inputs)
    x = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    x = tf.keras.layers.Add()([shortcut, x])
    
    x = tf.keras.layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

input = tf.keras.layers.Input(shape=(299, 299, 3))    

x = EntryFlowModule(input)

# Middle Flow is repeated 8 times
for _ in range(8):
    x = MiddleFlowModule(x)

x = ExitFlowModule(x)

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='Xception')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'xception_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'xception_layers.png')).show()

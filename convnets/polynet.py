import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

    
def PolyInceptionModule(inputs):
    x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)

    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x2)

    shortshut = tf.keras.layers.Concatenate()([x1, x2])
    
    output = tf.keras.layers.Add()([inputs, shortshut])
    return output
        

input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)        

x = PolyInceptionModule(x)
x = PolyInceptionModule(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = PolyInceptionModule(x)
x = PolyInceptionModule(x)
x = PolyInceptionModule(x)
x = PolyInceptionModule(x)
x = PolyInceptionModule(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = PolyInceptionModule(x)
x = PolyInceptionModule(x)

x = tf.keras.layers.GlobalAvgPool2D()(x)

x = tf.keras.layers.Dropout(0.4)(x)

# Units in last layer are 3 per rp dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='PolyNet')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'polynet_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'polynet_layers.png')).show()

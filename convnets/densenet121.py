import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os

k = 32 # growth rate

def DenseModule(inputs):
    x10 = tf.keras.layers.Conv2D(filters=4*k, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    m1 = tf.keras.layers.Concatenate()([x10, inputs])

    x20 = tf.keras.layers.Conv2D(filters=k, kernel_size=(3, 3), padding='same')(m1)
    output = tf.keras.layers.Concatenate()([x20, x10, inputs])
    return output

def TransitionModule(inputs, theta=0.5):
    x = tf.keras.layers.Conv2D(filters=inputs.shape[-1] * theta, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    output = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    return output

input = tf.keras.layers.Input(shape=(224, 224, 3))    
c0 = tf.keras.layers.Conv2D(filters=2*k, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(c0)

x = DenseModule(x) # 1
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 5
x = DenseModule(x) # 6

x = TransitionModule(x)

x = DenseModule(x) # 1
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 5
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 10
x = DenseModule(x)
x = DenseModule(x) # 12

x = TransitionModule(x)

x = DenseModule(x) # 1
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 5
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 10
x = DenseModule(x)
x = DenseModule(x) 
x = DenseModule(x) 
x = DenseModule(x)
x = DenseModule(x) # 15
x = DenseModule(x) 
x = DenseModule(x) 
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 20
x = DenseModule(x) 
x = DenseModule(x) 
x = DenseModule(x)
x = DenseModule(x) # 24

x = TransitionModule(x)

x = DenseModule(x) # 1
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 5
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x)
x = DenseModule(x) # 10
x = DenseModule(x)
x = DenseModule(x) 
x = DenseModule(x) 
x = DenseModule(x)
x = DenseModule(x) # 15
x = DenseModule(x) # 16

x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='DenseNet121')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'densenet121_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'densenet121_layers.png')).show()

import tensorflow as tf
import visualkeras
from keras_sequential_ascii import keras2ascii
import os


class SELayer(tf.keras.layers.Layer):
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=channel // reduction, activation='relu')
        self.fc1 = tf.keras.layers.Dense(units=channel, activation='sigmoid')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc(x)
        x = self.fc1(x)
        return tf.keras.layers.Multiply()([inputs, x])


def IdentityModule(inputs, filters):
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = SELayer(channel=filters[2], reduction=int(filters[0]/4))(x)

    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ProjectionModule(inputs, filters):
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=2)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=2)(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = SELayer(channel=filters[2], reduction=int(filters[0]/4))(x)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


input = tf.keras.layers.Input(shape=(224, 224, 3))    
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = ProjectionModule(x, filters=(64, 64, 256))
x = IdentityModule(x, filters=(64, 64, 256))
x = IdentityModule(x, filters=(64, 64, 256))

x = ProjectionModule(x, filters=(128, 128, 512))
x = IdentityModule(x, filters=(128, 128, 512))
x = IdentityModule(x, filters=(128, 128, 512))
x = IdentityModule(x, filters=(128, 128, 512))

x = ProjectionModule(x, filters=(256, 256, 1024))
x = IdentityModule(x, filters=(256, 256, 1024))
x = IdentityModule(x, filters=(256, 256, 1024))
x = IdentityModule(x, filters=(256, 256, 1024))
x = IdentityModule(x, filters=(256, 256, 1024))
x = IdentityModule(x, filters=(256, 256, 1024))

x = ProjectionModule(x, filters=(512, 512, 2048))
x = IdentityModule(x, filters=(512, 512, 2048))
x = IdentityModule(x, filters=(512, 512, 2048))

x = tf.keras.layers.GlobalAvgPool2D()(x)

# Units in last layer are 3 per rps dataset
output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=output, name='SENet-ResNet-50')

if __name__ == '__main__':
    model.summary()
    tf.keras.utils.plot_model(model, os.path.join('architectures', 'senet_resnet50_model.png'), show_shapes=True)
    # keras2ascii(model)
    visualkeras.layered_view(model, legend=True, to_file=os.path.join('architectures', 'senet_resnet50_layers.png')).show()

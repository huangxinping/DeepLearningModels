import tensorflow as tf
import os
from convnets.alexnet import model as alexnet
from convnets.zfnet import model as zfnet
from convnets.lenet5 import model as lenet5
from convnets.vgg16 import model as vgg16
from convnets.vgg19 import model as vgg19
from convnets.googlenet import model as googlenet
from convnets.resnet18 import model as resnet18
from convnets.resnet34 import model as resnet34
from convnets.resnet50 import model as resnet50
from convnets.resnet101 import model as resnet101
from convnets.resnet152 import model as resnet152
from convnets.densenet121 import model as densenet121
from convnets.resnext50 import model as resnext50
from convnets.xception import model as xception
from convnets.mobilenetv1 import model as mobilenetv1
from convnets.mobilenetv2 import model as mobilenetv2
from convnets.squeezenet import model as squeezenet
from convnets.polynet import model as polynet
from convnets.shufflenet import model as shufflenet
from convnets.senet import model as senet
from convnets.regnet import model as regnet
from convnets.convmixer import model as convmixer
import click


def create_generator(input_shape=(244, 224)):
    training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest'
    )

    train_generator = training_datagen.flow_from_directory(
        'datasets/rps/train',
        target_size=input_shape,
        class_mode='categorical',
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'datasets/rps/test',
        target_size=input_shape,
        class_mode='categorical'
    )
    return train_generator, validation_generator


@click.command()
@click.option("--target_size", default=224)
@click.option("--name", default="alexnet")
def main(name, target_size):
    if name == 'lenet5':
        model = lenet5
    elif name == 'alexnet':
        model = alexnet
    elif name == 'zfnet':
        model = zfnet
    elif name == 'vgg16':
        model = vgg16
    elif name == 'vgg19':
        model = vgg19
    elif name == 'googlenet':
        model = googlenet
    elif name == 'resnet':
        model = resnet34
    elif name == 'densenet':
        model = densenet121
    elif name == 'resnext':
        model = resnext50
    elif name == 'xception':
        model = xception 
    elif name == 'mobilenet':
        model = mobilenetv2
    elif name == 'squeezenet':
        model = squeezenet
    elif name == 'shufflenet':
        model = shufflenet
    elif name == 'senet':
        model = senet
    elif name == 'regnet':
        model = regnet
    elif name == 'convmixer':
        model = convmixer
    else:
        model = alexnet
        
    if name == 'xception':  # The input shape is (299, 299, 3)
        target_size = 299
    train_generator, validation_generator = create_generator(input_shape=(target_size, target_size))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
    print(f'The {name} is training...')
    model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)
    model.save(os.path.join('models', f'{name}.h5'))


if __name__ == '__main__':
    main()

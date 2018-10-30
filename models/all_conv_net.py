#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, GlobalAveragePooling2D, BatchNormalization


def get_model(n_classes, activation='relu'):
    model = Sequential()

    model.add(Conv2D(96, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # --------------------------------------------
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # --------------------------------------------
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (1, 1), padding='valid'))
    model.add(Activation(activation))
    model.add(BatchNormalization())

    model.add(Conv2D(n_classes, (1, 1), padding='valid'))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model
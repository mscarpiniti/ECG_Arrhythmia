# -*- coding: utf-8 -*-
"""
This file defines the models to be used for the classification of arrhythmia by
using scalograms and phasograms of ECG signals in the MIT-BIH dataset, as proposed
in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", *Sensors*, Vol. 24, N. 24, Paper 8043, December 2024. 
DOI: https://doi.org/10.3390/s24248043.


Created on Mon Nov  6 15:47:19 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# TensorFlow ≥2.0 is required
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Rescaling, concatenate
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D



# Create the AlexNet-based S1 strategy: 1 channel only (scalogram)
def S1(LR=0.001):

    # Input definition
    in_m = Input(shape=(224, 224, 1))

    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    y_m = Flatten()(x_m)

    # Classifier
    y = Dense(4096, activation='relu')(y_m)
    y = Dropout(0.5)(y)

    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    output = Dense(5, activation='softmax')(y)


    # Model definition
    net = Model(in_m, output)

    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net




# Create the AlexNet-based S2 strategy: 1 channel only (phasogram)
def S2(LR=0.001):

    # Input definition
    in_m = Input(shape=(224, 224, 1))

    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    y_m = Flatten()(x_m)

    # Classifier
    y = Dense(4096, activation='relu')(y_m)
    y = Dropout(0.5)(y)

    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    output = Dense(5, activation='softmax')(y)


    # Model definition
    net = Model(in_m, output)

    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net




# Create the AlexNet-based S3 strategy: 2 channels early fusion (scalogram + phasogram)
def S3(LR=0.001):

    # Input definition
    in_mp = Input(shape=(224, 224, 2))

    x_mp = Rescaling(scale=1./255)(in_mp)

    x_mp = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_mp)
    x_mp = BatchNormalization()(x_mp)
    x_mp = MaxPool2D((3, 3), strides=(2, 2))(x_mp)

    x_mp = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_mp)
    x_mp = BatchNormalization()(x_mp)
    x_mp = MaxPool2D((3, 3), strides=(2, 2))(x_mp)

    x_mp = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_mp)
    x_mp = BatchNormalization()(x_mp)

    x_mp = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_mp)
    x_mp = BatchNormalization()(x_mp)

    x_mp = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_mp)
    x_mp = BatchNormalization()(x_mp)
    x_mp = MaxPool2D((3, 3), strides=(2, 2))(x_mp)

    y_mp = Flatten()(x_mp)

    # Classifier
    y = Dense(4096, activation='relu')(y_mp)
    y = Dropout(0.5)(y)

    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    output = Dense(5, activation='softmax')(y)


    # Model definition
    net = Model(in_mp, output)

    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net





# Create the AlexNet-based S4 strategy: intermediate data fusion in Conv layers
def S4(LR=0.001):

    # Input definition
    in_m = Input(shape=(224, 224, 1))
    in_p = Input(shape=(224, 224, 1))

    # First branch
    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    y_m = BatchNormalization()(x_m)


    # Second branch
    x_p = Rescaling(scale=1./255)(in_p)

    x_p = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    y_p = BatchNormalization()(x_p)


    # Concatenation (INTERMEDIATE DATA FUSION)
    y = concatenate([y_m, y_p], axis=3)

    y = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D((3, 3), strides=(2, 2))(y)


    # Classifier
    z = Flatten()(y)

    z = Dense(4096, activation='relu')(z)
    z = Dropout(0.5)(z)

    z = Dense(4096, activation='relu')(z)
    z = Dropout(0.5)(z)

    output = Dense(5, activation='softmax')(z)


    # Model definition
    net = Model([in_m, in_p], output)


    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net




# Create the AlexNet-based S5 strategy: intermediate data fusion before Dense layers
def S5(LR=0.001):

    # Input definition
    in_m = Input(shape=(224, 224, 1))
    in_p = Input(shape=(224, 224, 1))

    # First branch
    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    y_m = Flatten()(x_m)


    # Second branch
    x_p = Rescaling(scale=1./255)(in_p)

    x_p = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    y_p = Flatten()(x_p)


    # Concatenation (INTERMEDIATE DATA FUSION)
    y = concatenate([y_m, y_p], axis=1)


    # Classifier
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    output = Dense(5, activation='softmax')(y)


    # Model definition
    net = Model([in_m, in_p], output)


    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net




# Create the AlexNet-based S6 strategy: intermediate data fusion in the last Dense layer
def S6(LR=0.001):

    # Input definition
    in_m = Input(shape=(224, 224, 1))
    in_p = Input(shape=(224, 224, 1))

    # First branch
    x_m = Rescaling(scale=1./255)(in_m)

    x_m = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)

    x_m = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_m)
    x_m = BatchNormalization()(x_m)
    x_m = MaxPool2D((3, 3), strides=(2, 2))(x_m)

    y_m = Flatten()(x_m)
    y_m = Dense(4096, activation='relu')(y_m)


    # Second branch
    x_p = Rescaling(scale=1./255)(in_p)

    x_p = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(384, (3, 3), strides=(1,1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)

    x_p = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = MaxPool2D((3, 3), strides=(2, 2))(x_p)

    y_p = Flatten()(x_p)
    y_p = Dense(4096, activation='relu')(y_p)


    # Concatenation (INTERMEDIATE DATA FUSION)
    y = concatenate([y_m, y_p], axis=1)


    # Classifier
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)

    output = Dense(5, activation='softmax')(y)


    # Model definition
    net = Model([in_m, in_p], output)


    # Display the model's architecture
    # net.summary()

    net.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return net

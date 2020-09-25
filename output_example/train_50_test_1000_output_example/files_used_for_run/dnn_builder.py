###### Imports #########
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
import sklearn as skl
from sklearn import *
from mlxtend.data import loadlocal_mnist
import platform
from Dictionary_Learning.learn_dictionary import data_generator

###### Imports- DL ################
import datetime
import os
import shutil

from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np
from tensorflow import keras

from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense

from tensorflow.keras import layers

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPool2D

# from keras.layers.core import Activation
# from keras.layers.core import Dropout
# from keras.layers.core import Lambda
# from keras.layers.core import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input




def get_conv_layer(inputs, name):
    """
    Used to generate a default set of hidden layers. The structure used in this network is defined as:
    Conv2D -> BatchNormalization -> Pooling -> Dropout
    """
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(3, 3), dilation_rate=(1, 1), padding="same",
                activation="relu",name=name+"_conv1")(inputs)
    # x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), dilation_rate=(1, 1), padding="same",
    #            kernel_initializer='glorot_uniform', activation="relu",name=name+"_conv2")(x)

    return x

def get_FC_later(inputs, name):
    """
    Used to generate a default set of hidden layers. The structure used in this network is defined as:
    Conv2D -> BatchNormalization -> Pooling -> Dropout
    """
    x = Flatten(name=name + "_flatten1")(inputs)
    x = Dense(10, activation="softmax", name=name+"_Dense1")(x)
    return x


def get_model_regular_conv_Net():
    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)
    conv_layer= get_conv_layer(inputs=inputs, name="conv_layer")
    output_layer= get_FC_later(conv_layer, name="output")
    model = Model(inputs=inputs,
                  outputs=output_layer,
                  name="regular_conv_Net")
    return model

def get_model_dictionary_learning(code_vector_size):
    input_shape = (code_vector_size, 1)
    inputs = Input(shape=input_shape)
    output_layer= get_FC_later(inputs, name="output")
    model = Model(inputs=inputs,
                  outputs=output_layer,
                  name="dictionary_learning_Net")
    return model
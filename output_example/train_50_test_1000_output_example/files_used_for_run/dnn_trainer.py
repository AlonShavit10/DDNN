###### Imports - dictionary #########
from time import time
import matplotlib.pyplot as plt
import scipy as sp
import cv2
import sklearn as skl
from sklearn import *
import platform

###### Imports- DL ################
import os

from tensorflow.keras.utils import to_categorical

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPool2D

# from keras.layers.core import Activation
# from keras.layers.core import Dropout
# from keras.layers.core import Lambda
# from keras.layers.core import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import graphviz
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Layer
from contextlib import redirect_stdout
from Dictionary_Learning.learn_dictionary.utils import *
from Dictionary_Learning.learn_dictionary.dict_learner import *
from Dictionary_Learning.learn_dictionary.dnn_builder import *
from Dictionary_Learning.learn_dictionary.descriptors_finder import *
from Dictionary_Learning.learn_dictionary import parameters, data_generator

params=parameters.Parameters()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 0.38
session = tf.compat.v1.Session(config=config)
def train_model(base_folder,run_name):


    tf.compat.v1.keras.backend.set_session(session)
    class_train_images_array, class_test_images_array = get_train_test_images(params.images_train_per_class,params.images_test_per_calss)
    dict_atoms,dict_objects=run_dict_learning(class_train_images_array)

    show_img(class_train_images_array[0][1],"bla")

    ##############################################################

    data_and_gt_train,norm_min_array,norm_max_array,data_and_gt_test=get_code_representatoin_for_train_test_images(class_train_images_array,class_test_images_array,dict_atoms,dict_objects)

    ### create folders

    run_dir = (os.path.join(base_folder, run_name))

    model_dir = (os.path.join(run_dir, "saved_model"))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    specific_run_tb_log_folder = (os.path.join(run_dir, "tensorboard"))
    if not os.path.exists(specific_run_tb_log_folder):
        os.mkdir(specific_run_tb_log_folder)

    current_file_full = os.path.abspath(__file__)
    current_file_name = os.path.basename(__file__)
    # shutil.copy2(current_file_full,
    #              os.path.join(run_dir, current_file_name))
    ###

    """
    create generators
    """
    # Train
    gen_object_train = data_generator.DataGenerator(data_and_gt_train, norm_min_array, norm_max_array, mode=params.mode)
    train_generator= gen_object_train.generate_images(batch_size=params.batch, inf_looping=True)

    gen_object_test = data_generator.DataGenerator(data_and_gt_test, norm_min_array, norm_max_array, mode=params.mode)
    test_generator= gen_object_test.generate_images(batch_size=params.batch, inf_looping=True)
    """
    build networks
    """

    if params.mode=="img":
        model =  get_model_regular_conv_Net()
    else:
        model = get_model_dictionary_learning(get_code_vector_size(params.mode))

    # print summary
    keras.utils.plot_model(model, to_file=os.path.join(run_dir,"out.png"), show_shapes=True, show_layer_names=True, dpi=96 * 2,
                           expand_nested=True)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(params.lr),
        metrics=['accuracy'],
    )

    tbCallBack = TensorBoard(log_dir=specific_run_tb_log_folder, histogram_freq=1)
    callbacks = [tbCallBack]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=(params.images_train_per_class*10) // params.batch,
                                  epochs=params.epochs,
                                  callbacks=callbacks,
                                  validation_data=test_generator,
                                  validation_steps=max(params.images_test_per_calss*10 // params.batch, 1))

    # save model
    model_path = os.path.join(model_dir, "full_model.hdf5")
    model.save(model_path)


    print ("training is finished!")
    return model_path, data_and_gt_train, data_and_gt_test, norm_min_array, norm_max_array, params.mode
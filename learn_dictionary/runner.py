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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import graphviz
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from contextlib import redirect_stdout
# from keras.models import load_model
from Dictionary_Learning.learn_dictionary.mnist_loader import *
from Dictionary_Learning.learn_dictionary.dnn_trainer import *
from Dictionary_Learning.learn_dictionary import parameters
params=parameters.Parameters()

##############################################################################
print("****** Started: " + params.run_name +  "******" )
if params.script_mode=="train":
    print ("Trian and eval started...")

    run_dir = (os.path.join(params.base_folder, params.run_name))
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # copy configurations
    cur_base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files_dir = run_dir + "\\files_used_for_run"
    # if not os.path.exists(files_dir):
    #     os.mkdir(files_dir)
    try:
        shutil.copytree(cur_base_folder + '\\learn_dictionary',
                        files_dir)
    except:
        print("error copying directory")

    # train and save
    model_path, data_and_gt_train, data_and_gt_test, norm_min_array, norm_max_array, mode = train_model(params.base_folder,params.run_name)
    saved_data_file=os.path.join(os.path.dirname(os.path.abspath(model_path)), "saved_data.npz")
    np.savez_compressed(saved_data_file, data_and_gt_train=data_and_gt_train, data_and_gt_test=data_and_gt_test, norm_min_array=norm_min_array, norm_max_array=norm_max_array, mode=mode)

elif params.script_mode=="eval":
    print("Eval only started for " + str(params.path_for_eval))
    model_path= os.path.join(params.path_for_eval,"saved_model")
    saved_data_file = os.path.join(model_path, "saved_data.npz")
    model_path=model_path+"\\full_model.hdf5"
else:
    raise NameError


# load
def load_and_eval(model_path,saved_data_file, eval_mode):
    model = tf.keras.models.load_model(model_path)
    loaded_data = np.load(saved_data_file)

    loaded_data.allow_pickle=True
    data_and_gt_train = loaded_data['data_and_gt_train']
    data_and_gt_test = loaded_data['data_and_gt_test']
    norm_min_array = loaded_data['norm_min_array']
    norm_max_array = loaded_data['norm_max_array']
    mode = loaded_data['mode']


    if eval_mode=="test":
        gen_object_test = data_generator.DataGenerator(data_and_gt_test, norm_min_array, norm_max_array, mode=mode)
        generator = gen_object_test.generate_images(batch_size=1, inf_looping=False)
    elif eval_mode=="train":
        gen_object_train = data_generator.DataGenerator(data_and_gt_train, norm_min_array, norm_max_array, mode=mode)
        generator = gen_object_train.generate_images(batch_size=1, inf_looping=False)
    else:
        raise NameError

    # evaluate
    res_array=[[],[],[],[],[],[],[],[],[],[]]
    success = 0
    failed = 0
    for img, gt in generator:
        prediction = model.predict(img)
        predicted = np.argmax(prediction)
        gt_number = np.argmax(gt)

        res_array[int(gt_number)].append(int(predicted))

        if predicted==gt_number:
            success+=1
        else:
            failed+=1
        print("predicted: " + str(predicted ))
        print("GT: " + str(gt_number))
        print ("***")

    success_rate = success/ (success+failed)
    tot_samples = success+failed

    conf_matrix = create_confusion_matrix(res_array)

    # print ("total success: %s, total failure: %s, success value is: %s" %(str(success), str(failed), str(success_rate)))
    return tot_samples,success_rate,conf_matrix

print ("************************ evaluating train data... ************************")
tot_samples_train,success_rate_train,conf_matrix_train=load_and_eval(model_path,saved_data_file, eval_mode="train")
print ("************************ evaluating test data... ************************")
tot_samples_test,success_rate_test,conf_matrix_test=load_and_eval(model_path,saved_data_file, eval_mode="test")

print("************************")
print("total train samples: %s, total test samples: %s"% (str(tot_samples_train), str(tot_samples_test)))
print("total success for train: %s, total success for test: %s"% (str(success_rate_train), str(success_rate_test)))


conf_matrix_test_int=np.int32(conf_matrix_test)

conf_matrix_trian_int=np.int32(conf_matrix_train)

if params.script_mode=="train":
    round_train = round(success_rate_train*100)
    round_test=round(success_rate_test*100)
    os.rename(run_dir, run_dir+"_"+str(round_train)+"_"+str(round_test))
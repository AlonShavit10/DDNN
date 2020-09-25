import sklearn as skl
import numpy as np
from Dictionary_Learning.learn_dictionary import parameters
from Dictionary_Learning.learn_dictionary.descriptors_finder import *
from Dictionary_Learning.learn_dictionary.mnist_loader import *
params=parameters.Parameters()

def dict_from_matrix(matrix, atoms=10, alpha=1, n_iter=50):
    """
    :param matrix:  (tot_descriptors X tot features per descriptos)
    :param atoms: dim of  the dictionary
    :param alpha: sparsity control
    :param n_iter: tot iters for the calculation
    :return: dictionary atoms of shape : (atoms X tot features per descriptor)
    """

    dict_object= skl.decomposition.MiniBatchDictionaryLearning(n_components=atoms, alpha=alpha, n_iter=n_iter)
    dict_for_image_patches= dict_object.fit(matrix)
    dictionary_atoms = dict_for_image_patches.components_
    return dictionary_atoms, dict_object

def get_y_tag_from_descriptors_matrix(anchors_descriptors, dict:skl.decomposition.MiniBatchDictionaryLearning, V, transform_algorithm, **kwargs):
    """
    :param anchors_descriptors: matrix of shape (num_of_anchorsX128 )
    :param dict: dictionary learning object
    :param V: the whole dictionary (number_of_atoms X 128)
    :param transform_algorithm: 'omp' / 'lars'/ 'threshold'
    :param kwargs: {'transform_n_nonzero_coefs': 2} / for threshold: {'transform_alpha': .1}
    :return:
    """

    dict.set_params(transform_algorithm=transform_algorithm, **kwargs)
    anchor_coeffs = dict.transform(anchors_descriptors)
    # now anchor_coeffs is of shape (num_of_anchors X number_of_atoms)
    # V should be of shape (number_of_atoms X 128)
    anchor_rep_vector = np.dot(anchor_coeffs, V)
    # now anchor_rep_vector is of shape (num_of_anchors X 128)
    y_tag = anchor_rep_vector.sum(axis=0)
    return y_tag

# TODO: only y_tag is diff between iters. the rest no. should underasnd why!!!
def get_img_codes_representations(img,dict_atoms,dict_objects):
    y_tag, y_tag_2, y_tag_3, y_tag_4=get_image_code_representation(img,dict_atoms,dict_objects)
    y_1, y_2, y_3, y_4, y_12, y_123, y_1234 = get_different_representations(y_tag,y_tag_2,y_tag_3,y_tag_4)
    possible_representations = [y_1, y_2, y_3, y_4, y_12, y_123, y_1234]
    # possbile_rep_shapes =[y_1.__len__(), y_2.__len__(), y_3.__len__(), y_4.__len__(), y_12.__len__(), y_123.__len__(), y_1234.__len__()]


    max_vals = np.array([(np.max(y_1),np.max(y_2),np.max(y_3),np.max(y_4),np.max(y_12),np.max(y_123),np.max(y_1234))])
    min_vals = np.array([(np.min(y_1), np.min(y_2), np.min(y_3), np.min(y_4), np.min(y_12), np.min(y_123), np.min(y_1234))])
    return possible_representations, max_vals, min_vals


"""
different ways to represent the data
"""
def get_different_representations(y_tag,y_tag_2,y_tag_3,y_tag_4):
    # y_1= y_tag# 128 size vector
    # y_2=y_tag_2# 128 size vector
    # y_3=y_tag_3# 128 size vector
    # y_4=y_tag_4 # 128 size vector
    y_12= np.concatenate((y_tag,y_tag_2))# 256 size vector
    y_123=np.concatenate((y_tag,y_tag_2,y_tag_3)) # 384 size vector
    y_1234=np.concatenate((y_tag,y_tag_2,y_tag_3,y_tag_4)) # 512 size vector
    return y_tag,y_tag_2,y_tag_3,y_tag_4,y_12,y_123,y_1234

"""
Feature Coding Layers
"""
# TODO check here the anchors_descriptors. it's obioues sinve it doesnt not depend on the image!!! only y_tag depend!
# TODO need to decompose the image using the dict and then get y''...! now the achore is jus for fun!!!
def get_image_code_representation(img,dict_atoms,dict_objects):
    # First Feature Coding Layer
    spesific_img_descriptor_mat = get_sift_descriptors_from_img(img)

    if params.kwargs is not None:
        y_tag = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[0],
                                                  V=dict_atoms[0], transform_algorithm=params.transform_algorithm,  **params.kwargs)
    else:
        y_tag = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[0], V =dict_atoms[0], transform_algorithm =params.transform_algorithm
                                                  )
    # Second Feature Coding Layer:
    if params.kwargs is not None:
        y_tag_2 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[1], V =dict_atoms[1], transform_algorithm =params.transform_algorithm,
                                                  **params.kwargs)
    else:
        y_tag_2 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[1], V =dict_atoms[1], transform_algorithm =params.transform_algorithm
                                                  )
    # third Feature Coding Layer:
    if params.kwargs is not None:
        y_tag_3 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[2], V =dict_atoms[2], transform_algorithm =params.transform_algorithm,
                                                  **params.kwargs)
    else:
        y_tag_3 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[2], V =dict_atoms[2], transform_algorithm =params.transform_algorithm
                                                  )
    # Forth Feature Coding Layer:
    if params.kwargs is not None:
        y_tag_4 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[3], V =dict_atoms[3], transform_algorithm =params.transform_algorithm,
                                                  **params.kwargs)
    else:
        y_tag_4 = get_y_tag_from_descriptors_matrix(anchors_descriptors=spesific_img_descriptor_mat, dict=dict_objects[3], V =dict_atoms[3], transform_algorithm =params.transform_algorithm
                                                  )
    return y_tag,y_tag_2,y_tag_3,y_tag_4

def get_code_vector_size(mode):
    if mode=="1" or mode=="2" or mode=="3" or mode=="4":
        code_vector_size=128
    elif mode=="12":
        code_vector_size=256
    elif mode == "123":
        code_vector_size = 384
    elif mode == "1234":
        code_vector_size = 512
    else:
        raise NameError
    return code_vector_size

def run_dict_learning(class_train_images_array):
    """
    Data Preparation
    """

    """
    Feature Extraction Layer:
    """
    print ("Calculating Descriptors...")
    all_classes_descriptors_matrix=get_all_train_images_descriptors_matrix(class_train_images=class_train_images_array)
    """
    Dictionary Learning Layers:
    """
    print ("Started: Learning Dictionaries...")

    # First Dictionary Learning Layer:
    print("1/4")
    dictionary_atoms_1, dict_object_1 = dict_from_matrix(matrix=all_classes_descriptors_matrix, atoms=params.num_of_atoms_dict_1, alpha=params.alpha, n_iter=params.n_iters)
    # Second dictionary learning layer
    print("2/4")
    dictionary_atoms_2, dict_object_2 = dict_from_matrix(matrix=dictionary_atoms_1, atoms=params.num_of_atoms_dict_2, alpha=params.alpha, n_iter=params.n_iters)
    # third dictionary learning layer
    print("3/4")
    dictionary_atoms_3, dict_object_3 = dict_from_matrix(matrix=dictionary_atoms_2, atoms=params.num_of_atoms_dict_3, alpha=params.alpha, n_iter=params.n_iters)
    # forth dictionary learning layer
    print("4/4")
    dictionary_atoms_4, dict_object_4 = dict_from_matrix(matrix=dictionary_atoms_3, atoms=params.num_of_atoms_dict_4, alpha=params.alpha, n_iter=params.n_iters)

    dict_atoms = [dictionary_atoms_1,dictionary_atoms_2,dictionary_atoms_3,dictionary_atoms_4]
    dict_objects = [dict_object_1,dict_object_2,dict_object_3,dict_object_4]
    print ("Finished: Learning Dictionaries")

    return dict_atoms,dict_objects

def get_code_representatoin_for_train_test_images(class_train_images_array,class_test_images_array,dict_atoms,dict_objects):

    data_and_gt_train=[]

    print("Started: Learning images representations for train...")
    norm_max_array= -np.inf * np.ones((1, 7))
    norm_min_array= +np.inf * np.ones((1, 7))
    for class_num in range (0,10):
        class_imgages = class_train_images_array[class_num][:]
        for img in class_imgages:
            gt=get_one_hot(class_num)
            img_code, max_vals, min_vals =get_img_codes_representations(img,dict_atoms,dict_objects)
            data_and_gt_train.append(data_and_gt(image_code=img_code, image_regular=img, gt=gt))
            norm_max_array=np.max((norm_max_array, max_vals), axis=0)
            norm_min_array=np.min((norm_min_array, min_vals), axis=0)

    print("Finished: Learning images representations for train")

    data_and_gt_test=[]
    print("Started: Learning images representations for test...")
    for class_num in range (0,10):
        class_imgages = class_test_images_array[class_num][:]
        for img in class_imgages:
            gt=get_one_hot(class_num)
            img_code, max_vals, min_vals =get_img_codes_representations(img,dict_atoms,dict_objects)
            data_and_gt_test.append(data_and_gt(image_code=img_code, image_regular=img, gt=gt))
            norm_max_array=np.max((norm_max_array, max_vals), axis=0)
            norm_min_array=np.min((norm_min_array, min_vals), axis=0)
    print("Finished: Learning images representations for test")

    return data_and_gt_train,norm_min_array,norm_max_array,data_and_gt_test
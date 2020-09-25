import cv2
import numpy as np
from Dictionary_Learning.learn_dictionary import parameters

params=parameters.Parameters()

def get_sift_descriptors_from_img(img,tot_descriptors=None): #Higer:  less , more
    """

    :param img:
    :param tot_descriptors:
    :return: num_descriptors X 128
    """
    # img = class_train[1]
    sift = cv2.SIFT_create(nfeatures=tot_descriptors)

    anchor_points = np.zeros(img.shape)
    anchor_points[::params.ANCHOR_POINTS_STEP, ::params.ANCHOR_POINTS_STEP]=1
    # plt.figure()
    # plt.imshow(anchor_points)


    points = np.int32(np.argwhere(anchor_points > 0))
    keypoints = [cv2.KeyPoint(int(x[1]), int(x[0]), 1) for x in points]

    des = sift.compute(image=img,keypoints=keypoints)[1]#  des is a numpy array of shape num_of_keypoints X 128

    # des.shape

    # img_with_kp = cv2.drawKeypoints(img, keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.figure()
    # plt.imshow(img_with_kp)
    return des

def get_descriptors_for_multiple_images(images):
    """
    :param images:
    :param atoms:
    :param alpha:
    :param n_iter:
    :return: (tot
    """

    for i,img in enumerate(images):
        if i==0:
            images_descriptors = get_sift_descriptors_from_img(img)
        else:
            images_descriptors = np.concatenate((images_descriptors,get_sift_descriptors_from_img(img)))

    return images_descriptors

def get_descriptors_matrix_for_multiple_images(images):
    """
    :param images:
    :param atoms:
    :param alpha:
    :param n_iter:
    :return: (tot
    """
    images_descriptors_matrix =get_descriptors_for_multiple_images(images)
    # class_dict, dict_object = dict_from_matrix(matrix = images_descriptors_matrix, atoms=atoms, alpha=alpha, n_iter=n_iter)
    return images_descriptors_matrix

def get_all_train_images_descriptors_matrix (class_train_images):
    # images_descriptors_matrices = []
    all_classes_descriptors_matrix = np.empty((0, 128))
    for i in range(0, 10):
        images_descriptors_matrix = get_descriptors_matrix_for_multiple_images(class_train_images[i])
        # images_descriptors_matrices.append(images_descriptors_matrix)
        all_classes_descriptors_matrix = np.concatenate((all_classes_descriptors_matrix, images_descriptors_matrix))
    return all_classes_descriptors_matrix

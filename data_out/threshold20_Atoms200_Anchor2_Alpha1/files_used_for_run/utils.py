from matplotlib import pyplot as plt
import numpy as np

def show_img(img,name):
    plt.figure(name)
    plt.imshow(img,cmap="gray")


def create_confusion_matrix(res_array):
    conf_matrix = np.zeros((10,10))
    for gt_col in range(0,10):
        for predicted in res_array[gt_col]:
            conf_matrix[predicted,gt_col]+=1
    return conf_matrix
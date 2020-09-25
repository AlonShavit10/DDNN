import numpy as np
from mlxtend.data import loadlocal_mnist
from Dictionary_Learning.learn_dictionary import parameters

params=parameters.Parameters()

def load_mnist():
    X, y = loadlocal_mnist(
        images_path=params.images_path,
        labels_path=params.labels_path)
    X = np.reshape(np.uint8(X),(X.shape[0],28,28))
    return X,y


def get_mnist_images(class_num, num_images_train, num_images_test, norm=False):
    X, y = load_mnist()
    # norm
    if norm:
        X = (X-np.mean(X))/np.std(X)
    else:
        X=np.uint8(X)
        # X = (np.float32(X))/255 # 0 to 1 norm
    class_images = X[y==class_num]
    class_train = class_images[0:num_images_train]
    class_test = class_images[num_images_train:num_images_train+num_images_test]
    return class_train,class_test


def get_train_test_images(num_images_train,num_images_test):
    class_train_images_array = []
    class_test_images_array = []
    for i in range(0, 10):
        class_train_images, class_test_images = get_mnist_images(class_num=i, num_images_train=num_images_train, num_images_test=num_images_test)
        class_train_images_array.append(class_train_images)
        class_test_images_array.append(class_test_images)
    return class_train_images_array, class_test_images_array

def get_one_hot(class_num, nb_classes=10):
    class_num_np = np.int32(class_num)
    res = np.eye(nb_classes)[np.array(class_num_np).reshape(-1)]
    return np.float32(res.reshape(list(class_num_np.shape)+[nb_classes]))

class data_and_gt():
    def __init__(self,image_code,image_regular,gt):
        self.image_code=image_code
        self.image_regular = image_regular
        self.gt=gt
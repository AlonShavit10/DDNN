from matplotlib import pyplot as plt

def show_img(img,name):
    plt.figure(name)
    plt.imshow(img,cmap="gray")
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from reseau_de_neurone import ReseauDeNeurone

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def display_image_matrix(index, dataset='train'):
    if dataset == 'train':
        image = x_train[index]
    else:
        image = x_test[index]
    
    print("Matrice de pixels de l'image:")
    print(image)
    plt.imshow(image, cmap='gray')
    plt.show()

#display_image_matrix(0, dataset='train')

reseau = ReseauDeNeurone(x_train[0], 128, 128, 10)
reseau.forward()
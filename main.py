import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from reseau_de_neurone import ReseauDeNeurone
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

couche1, couche2, sortie = 128, 128, 10
epochs = 30
batch_size = 5000
min_perte = 0.3

reseau = ReseauDeNeurone(x_train[0], couche1, couche2, sortie, y_train[0])
if os.path.exists("modele_mnist.npz"):
    reseau.load_model("modele_mnist.npz")

for epoch in range(epochs):
    total_loss = 0
    indices = np.random.choice(len(x_train), batch_size, replace=False)
    x_batch = x_train[indices]
    y_batch = y_train[indices]
    for i in range(batch_size):
        reseau.image = x_batch[i]
        reseau.vWant = [0] * 10
        reseau.vWant[y_batch[i]] = 1
        loss = reseau.forward()
        reseau.backPropagation()
        total_loss += loss

    print(f"Epoch {epoch+1}/{epochs} - Perte: {total_loss / batch_size:.4f}")

    if (total_loss / batch_size) < min_perte:
        min_perte = (total_loss / batch_size)
        reseau.save_model("modele_mnist.npz")



index = np.random.randint(0, 10000)
image_test = x_test[index]
prediction = reseau.predict(image_test)

print(f"Prédiction du modèle: {prediction}, Valeur réelle: {y_test[index]}")

# Affichage de l'image testée
plt.imshow(image_test, cmap='gray')
plt.title(f"Prédiction: {prediction}, Réel: {y_test[index]}")
plt.show()

# def display_image_matrix(index, dataset='train'):
#     if dataset == 'train':
#         image = x_train[index]
#     else:
#         image = x_test[index]
    
#     print("Matrice de pixels de l'image:")
#     print(image)
#     plt.imshow(image, cmap='gray')
#     plt.show()

# #display_image_matrix(0, dataset='train')

# reseau = ReseauDeNeurone(x_train[0], 128, 128, 10, y_train[0])
# reseau.train()
# print(reseau.predict(x_train[0]))
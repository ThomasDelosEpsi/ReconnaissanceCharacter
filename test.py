import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt

# Charger le modèle entraîné
model = load_model('mnist_model.h5')

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les images de test
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255

# Fonction pour tester le modèle sur les premières images
def test_model(num_images=5):
    for i in range(num_images):
        # Prédire le chiffre
        prediction = model.predict(x_test[i].reshape(1, 784))
        predicted_label = np.argmax(prediction)

        # Afficher l'image, la prédiction et la véritable étiquette
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Prediction: {predicted_label}\nAttendu: {y_test[i]}")
        plt.axis('off')

    plt.show()

# Tester le modèle sur les 5 premières images
test_model(num_images=5)

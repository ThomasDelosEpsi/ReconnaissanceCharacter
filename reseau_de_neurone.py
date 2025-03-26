import random as rm
import numpy as np

class ReseauDeNeurone:
    def __init__(self, image, couche1, couche2, sortie, vWant):
        self.image = image
        self.couche1 = couche1
        self.couche2 = couche2
        self.sortie = sortie
        self.vWant = [0] * 10
        self.vWant[vWant-1] = 1
        self.entrer = []
        self.vCouche1 = []
        self.vCouche2 = []
        self.vSortie = []
        self.biasCouche1 = np.random.uniform(-1, 1, couche1)
        self.biasCouche2 = np.random.uniform(-1, 1, couche2)
        self.poid1 = np.random.uniform(-1, 1, (len(image) * len(image), couche1))
        self.poid2 = np.random.uniform(-1, 1, (couche1, couche2))
        self.poid3 = np.random.uniform(-1, 1, (couche2, sortie))
        self.error = []
        self.learning_rate = 0.01

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def start(self):
        self.entrer = [1 if pixel > 0 else 0 for row in self.image for pixel in row]

    def calculate(self, couche, poid, bias=None):
        if bias is not None:
            return self.sigmoid(np.dot(couche, poid) + bias)
        return self.sigmoid(np.dot(couche, poid))

    
    def forward(self):
        self.start()
        self.vCouche1 = self.calculate(self.entrer, self.poid1, self.biasCouche1)
        self.vCouche2 = self.calculate(self.vCouche1, self.poid2, self.biasCouche2)
        self.vSortie = self.calculate(self.vCouche2, self.poid3, None)
        self.error = self.vWant - self.vSortie
        return np.sum(self.error**2)


    def backPropagation(self):
        erreur_sortie = self.error * self.sigmoid(self.vSortie, deriv=True)
        erreur_couche2 = np.dot(erreur_sortie, self.poid3.T) * self.sigmoid(self.vCouche2, deriv=True)
        erreur_couche1 = np.dot(erreur_couche2, self.poid2.T) * self.sigmoid(self.vCouche1, deriv=True)

        self.poid3 += self.learning_rate * np.dot(self.vCouche2.reshape(-1,1), erreur_sortie.reshape(1,-1))
        self.poid2 += self.learning_rate * np.dot(self.vCouche1.reshape(-1,1), erreur_couche2.reshape(1,-1))
        self.poid1 += self.learning_rate * np.dot(np.array(self.entrer).reshape(-1,1), erreur_couche1.reshape(1,-1))

        self.biasCouche2 += self.learning_rate * erreur_couche2
        self.biasCouche1 += self.learning_rate * erreur_couche1

    def train(self, epochs=1000):
        for _ in range(epochs):
            self.forward()
            self.backPropagation()

    def predict(self, image):
        self.image = image
        self.forward()
        probabilities = self.softmax(self.vSortie) * 100
        return {i: round(prob, 2) for i, prob in enumerate(probabilities)}

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def save_model(self, filename="model_weights.npz"):
        np.savez(filename, poid1=self.poid1, poid2=self.poid2, poid3=self.poid3, 
                 biasCouche1=self.biasCouche1, biasCouche2=self.biasCouche2)
        print(f"Modèle sauvegardé dans {filename}")
    
    def load_model(self, filename="model_weights.npz"):
        data = np.load(filename)
        self.poid1 = data["poid1"]
        self.poid2 = data["poid2"]
        self.poid3 = data["poid3"]
        self.biasCouche1 = data["biasCouche1"]
        self.biasCouche2 = data["biasCouche2"]
        print(f"Modèle chargé depuis {filename}")


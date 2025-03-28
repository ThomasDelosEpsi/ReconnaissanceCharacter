import random as rm
import numpy as np
class ReseauDeNeurone:
    def __init__(self, image, couche1, couche2, sortie, vWant):
        self.image = image
        self.couche1 = couche1
        self.couche2 = couche2
        self.sortie = sortie
        self.vWant = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.vWant[vWant-1] = 1
        self.entrer = []
        self.vCouche1 = [] 
        self.vCouche2 = [] 
        self.vSortie = []
        self.biasCouche1 = []
        self.biasCouche2 = []
        self.poid1 = []
        self.poid2 = []
        self.poid3 = []
        self.error = []
        
        for i in range(0, couche1):
            self.biasCouche1.append(np.random.uniform(-1, 1))
        for i in range(0, couche2):
            self.biasCouche2.append(np.random.uniform(-1, 1))
        for i in range(0, (len(self.image)*len(self.image))):
            tmp = []
            for j in range(0, couche1):
                tmp.append(np.random.uniform(-1, 1))
            self.poid1.append(tmp)
        for i in range(0, couche1):
            tmp = []
            for j in range(0, couche2):
                tmp.append(np.random.uniform(-1, 1))
            self.poid2.append(tmp)
        for i in range(0, couche2):
            tmp = []
            for j in range(0, sortie):
                tmp.append(np.random.uniform(-1, 1))
            self.poid3.append(tmp)


    def sigmoid(self, x, arg = 0):
        a = 1 / (1 + np.exp(-1*x))

        if arg == 1 :
            a = a*(1-a)
        
        return a
       
        
    def start(self):
        for i in range(0, len(self.image)):
            for j in range(0, len(self.image[i])):
                if(self.image[i][j] > 0):
                    self.entrer.append(1)
                else:
                    self.entrer.append(0)

    def calculate(self,couche,Vcouche,poid,bias=None):
        for i in range(0, len(poid)):
            couche[i] = self.sigmoid(couche[i])

        for i in range(0, len(poid[0])):
            a1 = 0
            for j in range(0, len(poid)):
                a1 += couche[j] * poid[j][i]
            a1 = a1 / len(couche)
            Vcouche.append(a1)
            
        print(len(Vcouche))
        for i in range(0, len(Vcouche)):
            if(bias != None):
                a1 = Vcouche[i] + bias[i]
            else:
                a1 = Vcouche[i]
            if a1 > 1 :
                a1 = 1
            elif a1 < -1 :
                a1 = -1
            Vcouche[i] = a1

    def calculError(self, poid, arg = 0):
        if arg == 1:
            for i in range(0, len(poid)):
                self.error.append(self.vWant[i] - poid[i])
        else :
            error = []
            for i in range(0, len(self.error)):
                tmp = []
                for j in range(0, len(poid)):
                    tmp.append(self.error[i] * poid[j][i])
                error.append(tmp)
            return error

    def backPropagation(self, poid):
        error = self.calculError(poid)

    def forward(self):
        self.start()
        self.calculate(self.entrer,self.vCouche1,self.poid1,self.biasCouche1)
        self.calculate(self.vCouche1,self.vCouche2,self.poid2,self.biasCouche2)
        self.calculate(self.vCouche2,self.vSortie,self.poid3)
        self.calculError(self.vSortie, 1)
        print(self.error)
        print(self.vSortie)

                
              
                
from numpy import random, array, dot 
import numpy as np 
import matplotlib.pyplot as plt 
from random import choice 
import math 
import sympy 
class adalineANN(object): 
    def init(self, gamma=.01, trials=500, errors=[], weights=[]): 
        self.gamma = gamma 
        self.trials = trials 
        self.errors = errors 
        self.weights = weights 
    def train(self): 
        self.weights = random.rand(3) coordinates_class1 = [] coordinates_class2 = [] 
        for x in np.random.normal(2, .5, 20): 
            for y in np.random.normal(3, .5, 20): coordinates_class1.append(([x, y, 1], 1)) 
            break for x in np.random.normal(2, .25, 20): 
        for y in np.random.normal(-1, .25, 20): coordinates_class2.append(([x, y, 1], -1)) 
        break trainingData = coordinates_class1 + coordinates_class2 
    for i in range(self.trials):
         x, target = choice(trainingData) 
         y = np.dot(x, self.weights) error, errors = [], [] error = (target - y) 
         self.errors.append(error) for i in range(0, 3): 
             self.weights[i] += self.gamma * x[i] * (target - y) #????* 
             (sympy.cosh(y)**(-1))**2 
    def plot(self): 
        plt.plot(self.errors) 
        plt.show()
        A = adalineANN() 
        A.train() 
        A.plot()
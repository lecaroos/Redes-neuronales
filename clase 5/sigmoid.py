import numpy as np
import pytest
import random
import math

class Sigmoid:
    
    def __init__(self,n_inputs, learning_rate=0.1):
        #Initializes de perceptron depending on the number of inputs
        self.b = random.uniform(-2,2)          
        self.w = []
        for i in range(n_inputs):
            a = random.uniform(-2,2)
            self.w.append(a)
        self.lr = learning_rate
    
    def feed(self, input):
        #Gives the array with binary inputs to the perceptron.

        a = np.asmatrix(input)
        b = np.asmatrix(self.w).transpose()
        z = a.dot(b)[0,0] - self.b
        r = 1/(1+np.exp(-z))

        return r

    def train(self, input, desired_output):
        r = self.feed(input)
        diff = desired_output-r
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + (self.lr*input[i]*diff)
        self.b = self.b +(self.lr*diff)
        return r

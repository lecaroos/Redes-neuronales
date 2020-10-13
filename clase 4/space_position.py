import random
import matplotlib.pyplot as plt
import learning_perceptron
import numpy as np

def rand_point():
    return [random.randint(-100,100),random.randint(-100,100)]

def classify(point):
    if point[0]<=point[1]:
        return 1
    elif point[0]>point[1]:
        return 0

def create_set(n):
    input = []
    output = []
    for i in range(n):
        p = rand_point()
        c = classify(p)
        input.append(p)
        output.append(c)
    return [input,output]

def training(n_traning, n_test):
    set = create_set(n_traning)
    inputs = set[0]
    d_outputs = set[1]
    output_zeros = []
    output_ones = []
    p = learning_perceptron.Perceptron(2,learning_rate=0.1)

    

    for i in range(n_traning):
        p.train(inputs[i],d_outputs[i])

    set = create_set(n_test)
    inputs = set[0]

    for i in range(n_test):
        r = p.feed(inputs[i])
        if r == 1:
            output_ones.append(inputs[i])
        else :
            output_zeros.append(inputs[i])

    output_ones = np.asmatrix(output_ones).transpose().tolist()
    plt.scatter(output_ones[0],output_ones[1], c='lightblue')

    output_zeros = np.asmatrix(output_zeros).transpose().tolist()
    plt.scatter(output_zeros[0],output_zeros[1], c='coral')

    plt.plot([-100,100],[-100,100])
    plt.show()


training(30,500)


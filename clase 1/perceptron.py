import numpy
import pytest

class Perceptron:
    
    def __init__(self, bias, weights):
        #The bias is a Double, the wieght is an array and the lear.
        self.b = bias          
        self.w = weights
    
    def feed(self, input):
        #Gives the array with binary inputs to the perceptron.

        a = numpy.asmatrix(input)
        b = numpy.asmatrix(self.w).transpose()
        r = a.dot(b)[0,0]
        if (r+self.b)<=0:
            return 0
        else:
            return 1
        

class SumGate:

    def __init__(self):
        #Initilizes a perceptron with the NAND behaviour
        self.p = Perceptron(3,[-2,-2])
    
    def sum(self,x1,x2):
        #Adds 2 bits, returns the result and the carry
        r1 = self.p.feed([x1,x2])
        r2 = self.p.feed([x1,r1])
        r3 = self.p.feed([x2,r1])
        r = self.p.feed([r2,r3])
        carry = self.p.feed([r1,r1])

        return [r,carry]
    
         
#Inputs
a = [0,0]
b = [1,0]
c = [0,1]
d = [1,1]

def test_perceptron_or():
    p = Perceptron(-0.5,[1,1])
    assert p.feed(a) == 0,"Test Failed"
    assert p.feed(b) == 1,"Test Failed"
    assert p.feed(c) == 1,"Test Failed"
    assert p.feed(d) == 1,"Test Failed"

def test_perceptron_and():
    p = Perceptron(-1.5,[1,1])
    assert p.feed(a) == 0,"Test Failed"
    assert p.feed(b) == 0,"Test Failed"
    assert p.feed(c) == 0,"Test Failed"
    assert p.feed(d) == 1,"Test Failed"

def test_perceptron_nand():
    p = Perceptron(3,[-2,-2])
    assert p.feed(a) == 1,"Test Failed"
    assert p.feed(b) == 1,"Test Failed"
    assert p.feed(c) == 1,"Test Failed"
    assert p.feed(d) == 0,"Test Failed"

def test_SumGate():
    gate = SumGate()

    assert gate.sum(0,0) == [0,0], "Test Failed"
    assert gate.sum(1,0) == [1,0], "Test Failed"
    assert gate.sum(0,1) == [1,0], "Test Failed"
    assert gate.sum(1,1) == [0,1], "Test Failed"
    

test_perceptron_or()
test_perceptron_and()
test_perceptron_nand()
test_SumGate()

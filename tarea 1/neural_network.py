import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def sigmoid(z):
	return 1/(1 + np.exp(-z))


#Prepare data by spliting the table in 2
#The fisrt half is the inputs and the second half is the outputs
#Then we encode the outputs
#Finally we split the data into 2 sets using trian_test_split from the sklearn library
def prepare_data(dt_name):
  f = open(dt_name, "r")
  X = []
  aux = []
  y = []
  for i in f:
    line = i.split(",")
    l = len(line)
    x = line[:l-1]
    X.append(x)
    a = line[-1].rstrip()
    aux.append(a)
  for n, i in enumerate(X):
    for k, j in enumerate(i):
      X[n][k] = float(j)
  X = X[:len(X)-1]
  
  for i in range(len(aux)):
    if aux[i]=="Iris-setosa":
      y.append([1,0,0])
    elif aux[i]=="Iris-versicolor":
      y.append([0,1,0])
    elif aux[i]=="Iris-virginica":
      y.append([0,0,1])
  X = np.asmatrix(X)
  y = np.asmatrix(y)
  norm_X = minmax_scale(X)
  X_train, X_test, y_train, y_test = train_test_split(norm_X, y, test_size = 0.33, random_state = 42)
  return X_train, X_test, y_train, y_test


# Produce a neural network randomly initialized
def initialize_parameters(n_x, n_h, n_y):
	W1 = np.random.randn(n_h, n_x)
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h)
	b2 = np.zeros((n_y, 1))

	parameters = {
	"W1": W1,
	"b1" : b1,
	"W2": W2,
	"b2" : b2
	}
	return parameters

# Evaluate the neural network
def forward_prop(X, parameters):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  # Z value for Layer 1
  Z1 = np.dot(W1, X) + b1
  # Activation value for Layer 1
  A1 = np.tanh(Z1)
  # Z value for Layer 2
  Z2 = np.dot(W2, A1) + b2
  # Activation value for Layer 2
  A2 = sigmoid(Z2)

  cache = {
    "A1": A1,
    "A2": A2
  }
  return A2, cache

# Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y 
# We use the Mean Square Error cost function
def calculate_cost(A2, Y, m):
  # m is the number of examples
  A2 = np.asarray(A2)
  Y = np.asarray(Y)
  cost = np.sum((0.5 * (A2 - Y) ** 2).mean(axis=1))/m
  return cost

# Apply the backpropagation
def backward_prop(X, Y, cache, parameters, m):
  A1 = cache["A1"]
  A2 = cache["A2"]

  W2 = parameters["W2"]

  # Compute the difference between the predicted value and the real values
  dZ2 = A2 - Y
  dW2 = np.dot(dZ2, A1.T)/m
  db2 = np.sum(dZ2, axis=1)/m
  # Because d/dx tanh(x) = 1 - tanh^2(x)
  dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
  dW1 = np.dot(dZ1, X.T)/m
  db1 = np.sum(dZ1, axis=1)/m

  grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
  }

  return grads

# Third phase of the learning algorithm: update the weights and bias
def update_parameters(parameters, grads, learning_rate):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  dW1 = grads["dW1"]
  db1 = grads["db1"]
  dW2 = grads["dW2"]
  db2 = grads["db2"]

  W1 = W1 - learning_rate*dW1
  b1 = b1 - learning_rate*db1
  W2 = W2 - learning_rate*dW2
  b2 = b2 - learning_rate*db2
  
  new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
  }

  return new_parameters

# model is the main function to train a model
# X: is the set of training inputs
# Y: is the set of training outputs
# n_x: number of inputs (this value impacts how X is shaped)
# n_h: number of neurons in the hidden layer
# n_y: number of neurons in the output layer (this value impacts how Y is shaped)
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
  parameters = initialize_parameters(n_x, n_h, n_y)
  #m is the No. of training examples
  m = X.shape[1]

  for i in range(0, num_of_iters+1):
    rows, cols = X.shape
    cost = 0
    total_cost = 0
    total_cost_array = []

    for j in range(cols):
      x_j=X[:,j]
      y_j=Y[:,j]
      a2, cache = forward_prop(x_j, parameters)
      cost = calculate_cost(a2, y_j, m)
      total_cost+=cost
      total_cost_array.append(total_cost)
      grads = backward_prop(x_j, y_j, cache, parameters, m)
      parameters = update_parameters(parameters, grads, learning_rate)
    if(i%100 == 0):
      print('Cost after iteration# {:d}: {:f}'.format(i, cost))

  return parameters

# Make a prediction
# X: represents the inputs
# parameters: represents a model
# the result is the prediction
def predict(X, parameters):
  a2, cache = forward_prop(X, parameters)
  
  r = np.where(a2 == np.amax(a2))
  return r[0]





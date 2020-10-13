import neural_network as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable


#Makes the test Inputs go trough the NN and checks if it got the correct answer
#Then makes the confusion matrix as seen in class
def conf_matrix(X_test,y_test, model):
    rows_X, cols_X = X_test.shape
    y_true = []
    y_pred = []
    for i in range(cols_X):
        test = X_test[:,i]
        p = nn.predict(test, model)
        y_pred.append(int(p))
        r = y_test[:,i]
        if r[0]==1:
            y_true.append(0)
        elif r[1]==1:
            y_true.append(1)
        elif r[2]==1:
            y_true.append(2)   
    r = confusion_matrix(y_true,y_pred)
    r = np.asmatrix(r).transpose()
    return r


def main(data, epochs, lr):

    X_train, X_test, y_train, y_test = nn.prepare_data(data)
    X_train = np.asmatrix(X_train).transpose()
    y_train = np.asmatrix(y_train).transpose()
    X_test = np.asmatrix(X_test).transpose()
    y_test = np.asmatrix(y_test).transpose()
    rows, cols = X_train.shape
    n_x = rows
    n_h = 5
    n_y = 3
    model = nn.model(X_train, y_train,n_x,n_h,n_y,epochs,lr)
    r = conf_matrix(X_test,y_test,model)
    print(r)
    r = np.asarray(r)
    x = PrettyTable()
    x.field_names = ["   ","GoldLabel_Setosa", "GoldLabel_Versicolor","GoldLabel_Virginica" ]
    x.add_row(["Predicted_Setosa",r[0][0], r[0][1], r[0][2] ])
    x.add_row(["Predicted_Versicolor",r[1][0], r[1][1], r[1][2]])
    x.add_row(["Predicted_Virginica",r[2][0], r[2][1], r[2][2] ])
    print(x)
    

main(r"C:\Users\lecaroos\Documents\u\redes neuronales\proyecto\tarea 1\iris.data", 1000, 0.1)
import os
import sys
import math
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random

np.random.seed(100)

learning_rate = 0.03
epochs = 60

 
input_size, h1, h2, output_size = 5,10,10,2
# try to remove this znd use as variable - like declare them in main and pass to fucntons
weights = {'W1': np.random.randn( h1, input_size) * 1 / np.sqrt(input_size),
          'b1':np.random.randn( h1,1) * np.sqrt(1.0 / input_size),
          'W2':np.random.randn( h2, h1) * 1 / np.sqrt( h1),
          'b2':np.random.randn( h2,1) * np.sqrt(1.0/h1),
          'W3':np.random.randn( output_size, h2) * 1 / np.sqrt(h2),
          'b3':np.random.randn( output_size,1) * np.sqrt(1.0 / h2)
         }
print(weights)

#maybe change the variable names of derivatives
def backward(X, Y, A1,A2,A3,Z1,Z2,Z3):
    m = X.shape[1]
    
    dZ3 = A3 - Y
    
    dA2 = np.dot( weights['W3'].T, dZ3)
    dZ2 = signmoid_derivative(dA2, Z2)
    
    dA1 = np.dot( weights['W2'].T, dZ2)
    dZ1 = signmoid_derivative(dA1, Z1)
    
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
   
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    return dW3, db3, dW2, db2, dW1, db1
    
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#can change this - directly use nr inside return 
def softmax(x):
    nr = np.exp(x - x.max())
    return nr / np.sum(nr, axis=0)

def signmoid_derivative(dA, Z):
    return dA * sigmoid(Z) * (1 - sigmoid(Z))
    
    # forward pass
def forward(X):
    Z1 = np.dot( weights['W1'], X) +  weights['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot( weights['W2'], A1) +  weights['b2']
    A2 = sigmoid(Z2)
    Z3 = np.dot( weights['W3'], A2) +  weights['b3']
    A3 = softmax(Z3)
        
    return A1,A2,A3,Z1,Z2,Z3

def updateWeights(dW3, db3,dW2,db2,dW1,db1):
    weights['W1'] =  weights['W1'] - (learning_rate * dW1)
    weights['W2'] =  weights['W2'] - (learning_rate * dW2)
    weights['W3'] =  weights['W3'] - (learning_rate * dW3)
    weights['b1'] =  weights['b1'] - (learning_rate * db1)
    weights['b2'] =  weights['b2'] - (learning_rate * db2)
    weights['b3'] =  weights['b3'] - (learning_rate * db3)
    
def train(X, y):
    for i in range(epochs):
        for j in range(1,X.shape[1]+1):
            xm,ym=X[:,j-1:j],y[:,j-1:j]
            
            A1,A2,A3,Z1,Z2,Z3 =  forward(xm)
            dW3, db3, dW2, db2, dW1, db1 = backward(xm, ym, A1,A2,A3,Z1,Z2,Z3)
            updateWeights(dW3, db3, dW2, db2, dW1, db1)
            

def preprocess(X_train,Y_train, X_test):  
    # adding columns x1**2, x2**2 and x1*x2 for train data    
    X_train[2]=X_train[0]**2
    X_train[3]=X_train[1]**2
    X_train[4]=X_train[0]*X_train[1]
    
    # adding columns x1**2, x2**2 and x1*x2 for test data
    X_test[2]= X_test[0]**2
    X_test[3]= X_test[1]**2
    X_test[4]= X_test[0] * X_test[1]
    
    # matrices transpose
    trainX,trainY,testX = X_train.T,Y_train.T, X_test.T
    
    trainX,trainY,testX = trainX.values,trainY.values,testX.values
    
    # onehot encoding for train labels
    Y_onehot = np.zeros((trainY.size, trainY.max()+1))
    Y_onehot[np.arange(trainY.size), trainY] = 1
    Y_onehot =  Y_onehot.T

    return trainX, Y_onehot, testX
        
if __name__ == '__main__':
    
    #X_train = pd.read_csv(sys.argv[1], header=None)
    #Y_train = pd.read_csv(sys.argv[2], header=None)
    #X_test = pd.read_csv(sys.argv[3], header=None)
    '''

   
    X_train = pd.read_csv("public/xor_train_data.csv", header=None)
    Y_train = pd.read_csv("public/xor_train_label.csv", header=None)
    X_test = pd.read_csv("public/xor_test_data.csv", header=None)

    '''  
   
    X_train = pd.read_csv("public/circle_train_data.csv", header=None)
    Y_train = pd.read_csv("public/circle_train_label.csv", header=None)
    X_test = pd.read_csv("public/circle_test_data.csv", header=None)

    X_train = pd.read_csv("public/spiral_train_data.csv", header=None)
    Y_train = pd.read_csv("public/spiral_train_label.csv", header=None)
    X_test = pd.read_csv("public/spiral_test_data.csv", header=None)
    
        
        

    
    X=X_train.values
    y=Y_train.values
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)    
    
    
    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
    df=pd.DataFrame({'r1':r1.flatten(), 'r2':r2.flatten()})
    
    df[2]=df['r1']**2
    df[3]=df['r2']**2
    df[4]=df['r1']*df['r2']
    df=df.T
    grid=df.values

  
    trainX, Y_onehot,testX=preprocess(X_train.copy(),Y_train.copy(), X_test.copy())
  
    train(trainX, Y_onehot)
    
    # prediction
    A1,A2,output,Z1,Z2,Z3 = forward(testX)
    
    pred = np.argmax(output, axis=0)
    
    
    A1,A2,yhat,Z1,Z2,Z3 = forward(grid)
    yhat = np.argmax(yhat, axis=0)

    
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    
    
    pd.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
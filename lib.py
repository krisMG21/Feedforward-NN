import numpy as np
import time
import json
import os.path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def init_params(config):
    '''
    config: dict {
        "input_size": int,
        "hidden_size": int,
        "output_size": int
    }

    Initializes the matrices that will form the neural network
    * W1: First weights, from inputs to hidden layer
    * b1: First column of neurons, and their values
    * W2: Second weigths, to hidden to output
    * b2: Second column of biases
    '''
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    output_size = config["output_size"]

    if os.path.isfile('nn_weights.json'):                   #Values exists in a file from previous execution
        with open("nn_weights.json", "r") as json_file:     #and so we load them to keep training and making progress
            data = json.load(json_file)
        W1 = np.array(data["W1"])
        b1 = np.array(data["b1"])
        W2 = np.array(data["W2"])
        b2 = np.array(data["b2"])
        
    else:

        #We initialize with small values to push the accurracy improvement a bit further
        W1 = np.random.normal(size=(hidden_size, input_size)) * np.sqrt(1./(input_size))
        b1 = np.random.normal(size=(hidden_size, 1)) * np.sqrt(1./hidden_size) 
        W2 = np.random.normal(size=(output_size, hidden_size)) * np.sqrt(1./hidden_size)
        b2 = np.random.normal(size=(output_size, 1)) * np.sqrt(1./(input_size))
    
    return W1, b1, W2, b2
    
def ReLU(Z):
    '''
    Z[int, int] --> [int,int]  (in this program) \n
    Returns Z if positive, 0 if negative, used for activation, 
    we get 1 and 0 in booleans
    '''
    return np.maximum(Z, 0)

def softmax(Z):
    '''
    Z[int, int] --> A[int, int] \n
    Reduces the range of the inputs to a set of values of mean 0 and variance 1
    using the following ecuation.
    '''
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    '''
    W1,b1,W2,b2,X: [int,int] --> Z1,A1,Z2,A2: [int,int] \n
    Propagates the input values (X) through the net, returning
    the values required for back-propagation, the next step
    '''
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    '''
    Z:[int,int] --> [int,int] \n
    Derivative of the ReLu function, turns the negatives (and 0) into 0 and positives into 1
    '''
    return Z > 0

def one_hot(Y):
    '''
    Y: [int, int] --> [int,int] \n
    Returns a 0 filled vector, with positions corresponding to Y set to 1
    '''
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, A2, W2, X, Y, m):
    '''
    Z1,A1,A2...,W2,X,Y: [int,int] --> dW1: [10, 784], db1: [10,1], dW2: [10,10], db2: [10,1]
    Checks if the predictions (A2), correspond to the labels (Y), and fixes weights 
    backwards accordingly
    '''
    one_hot_Y = one_hot(Y)      
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    '''
    W1,b1,... dW2,db2: [int,int], alpha: float --> W1,b1,W2,b2: [int,int]\n
    Adjusts the weights and biases, using the dWi and dbi gotten on backwards prop
    using alpha factor, arbitrary and determined when calling
    '''
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    '''
    A2[int,int] --> [int]
    Returns all predictions for the input cases
    '''
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    '''
    [int], [int] --> acc: float
    Given the predictions and the real values, computes the accuracy
    of the lot
    '''
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, m, config):
    '''
    X,Y: [int,int],
    alpha: float,
    iterations:int
    m:int
    config: dict
    --> W1,b1,W2,b2: [int,int]

    Represents all the proccess of training when feeding the table of data to the network.
    For each iteration, all the rows are feeded as input and all the steps are made for getting
    the predictions and adjust the weights accordingly. 
    For feedback, each 10 iterations some info is given, such as the predictions and their corresponding values,
    as well as the current accuracy '''

    W1, b1, W2, b2 = init_params(config)
    for i in range(iterations):
        Z1, A1, _, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0: #Printing intervals
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))

    return W1, b1, W2, b2

def paint_number(index, X):
    '''
    index: int, X: [int, int] --> null
    Given X dataset and an index, prints on screen the number that it is stored
    '''
    current_image = X[:, index, None]
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)  # Show the plot without blocking, and wait for user input
    plt.pause(0.5)  # Pause to allow the plot to update
    time.sleep(1)  # Display for 5 seconds (or adjust as needed)
    plt.close()  # Close the plot

def make_predictions(X, W1, b1, W2, b2):
    '''
    X,W1...b2: [int,int] --> predictions[int]
    Given all the inputs (X), and the state of the network, returns the list
    of all the predictions
    '''
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, X_train, Y_train):
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    paint_number(index,X_train)

def show_fails(W1, b1, W2, b2, X_train, Y_train):
    index = 0
    fails = 0
    for _ in X_train:
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        
        if prediction != label:
            print("Prediction: ", prediction)
            print("Label: ", label)
            paint_number(index,X_train)

            fails += 1

        index +=1
    print(f"Total of wrong predictions: {fails}")


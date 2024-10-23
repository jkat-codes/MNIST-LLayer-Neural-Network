import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import os
from activations import *
import pickle
import tqdm
import itertools
import sys

## Hyperparameters go here ##
learning_rates = [0.2, 0.4, 0.6, 0.8, 0.09, 0.07, 0.05, 0.03]
best_accuracy = 0
best_hyperparams = {}

## Load the training data ##
csv = pd.read_csv("data/train.csv")
csv.head
data = np.array(csv)
m, n  = data.shape
# m is the amount of rows or images
# n is the amount of columns or pixels per image, -1 because the first column is the label that specifies which number is which
# development set / we don't train on this / we use for hyperparameters testing
np.random.shuffle(data)

# m is the number of training samples

data_dev = data[0:1000].T # first 1000 examples now converted from (m, n) to (n, m) each column is now an example, with each row being the corresponding pixel
Y_dev = data_dev[0] # this is the first column of the transposed data, meaning this is the column of all of the labels
X_dev = data_dev[1:] / 255 #this is every column, excluding the first since it has now become the 


data_train = data[1000:m].T
Y_train = data_train[0] ## this is the first column of the transposed data, meaning this is the column of all of the labels
X_train = data_train[1:] / 255
print(X_train[:, 0].shape) ## this is the number of pixels PER image
num_train_examples = np.squeeze(X_train[0].shape)

print("Number of nodes in input layer: ", X_train.shape[1])


def initialize_params(layer_dims): 
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L): 
        parameters[f'W{l}'] = np.random.rand(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2./layer_dims[l-1]) ## this coefficient is He initialization
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1)) 

    return parameters

layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
params = initialize_params(layers)
for l in range(1, len(layers)):
    print("Shape of W" + str(l) + ":", params['W' + str(l)].shape)
    print("Shape of B" + str(l) + ":", params['b' + str(l)].shape, "\n")



def linear_forward(A, W, b): 
    # this does Z = np.dot(A, W) + b
    Z = np.dot(A, W) + b

    return Z

def forward_prop(X, parameters, activation): 
    forward_cache = {}
    L = len(parameters) // 2 # divide by 2 because we have w and b
    
    forward_cache['A0'] = X

    for l in range(1, L):
        forward_cache['Z' + str(l)] = parameters['W' + str(l)].dot(forward_cache['A' + str(l-1)]) + parameters['b' + str(l)]
        
        if activation == 'tanh':
            forward_cache['A' + str(l)] = tanh(forward_cache['Z' + str(l)])
        else:
            forward_cache['A' + str(l)] = relu(forward_cache['Z' + str(l)])
            

    forward_cache['Z' + str(L)] = parameters['W' + str(L)].dot(forward_cache['A' + str(L-1)]) + parameters['b' + str(L)]
    forward_cache['A' + str(L)] = softmax(forward_cache['Z' + str(L)])
    
    return forward_cache['A' + str(L)], forward_cache

## test forward prop 
def test_forward_prop(): 
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)  
    aL, forward_cache = forward_prop(X_train, params, 'relu')

    for l in range(len(params) // 2 + 1): 
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)
        

def back_prop(AL, Y, parameters, forward_cache, activation): 
    grads = {}
    L = len(parameters) // 2 ## we divide by 2 because we have W and b
    m = AL.shape[1]

    grads["dZ" + str(L)] = AL - Y
    grads["dW" + str(L)] = 1. / m * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L - 1)].T)
    grads["db" + str(L)] = 1. / m * np.sum(grads["dZ" + str(L)], axis=1, keepdims=True)

    for l in reversed(range(1, L)): 
        if activation == "relu": 
            grads["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]) * (forward_cache["A" + str(l)] > 0) ## this is the derivative of relu
        elif activation == "tanh": 
            grads["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]) * deriv_tanh(forward_cache["A" + str(l)]) 
        
        grads["dW" + str(l)] = 1. / m * np.dot(grads["dZ" + str(l)], forward_cache["A" + str(l - 1)].T)
        grads["db" + str(l)] = 1. / m * np.sum(grads["dZ" + str(l)], axis=1, keepdims=True)

    return grads

## test back prop and observe shapes
def test_back_prop(): 
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)  
    aL, forward_cache = forward_prop(X_train, params, 'relu')

    for l in range(len(params) // 2 + 1): 
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)

    grads = back_prop(forward_cache["A" + str(3)], Y_train, params, forward_cache, 'relu')

    print('\n')

    for l in reversed(range(1, len(grads) // 3 + 1)): 
        print("Shape of dZ" + str(l) + " :", grads['dZ' + str(l)].shape)
        print("Shape of dW" + str(l) + " :", grads['dW' + str(l)].shape)
        print("Shape of dB" + str(l) + " :", grads['db' + str(l)].shape, "\n")

# update parameters so the network can learn
def update_parameters(parameters, grads, learning_rate): 
    L = len(parameters) // 2 ## we divide by 2 because we have W and b

    for l in range(L): 
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

# predictions function
def predict(X, Y, parameters, activation): 
    m = X.shape[1]
    y_prediction, caches = forward_prop(X, parameters, activation)

    if Y.shape[0] == 1: # this is for binary classification
        y_prediction = np.array(y_prediction > 0.5, dtype='float')
    else: # this is for multi-class classification
        y_prediction = np.argmax(y_prediction, axis=0)
        Y = np.argmax(Y, axis=0)
        accuracy = np.mean(y_prediction == Y)
        return accuracy

    return np.sum((y_prediction == Y) / m)

def model(X, Y, layer_dims, learning_rate=0.005, activation='relu', num_iterations=5000): 
    costs = []

    parameters = initialize_params(layer_dims)

    for i in tqdm.tqdm(range(0, num_iterations)): 
        AL, forward_cache = forward_prop(X, parameters, activation)

        cost = -np.mean(Y * np.log(AL + 1e-8)) ## cost function, we add 1e-8 in order to avoid log(0)

        grads = back_prop(AL, Y, parameters, forward_cache, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % (num_iterations / 10) == 0: 
            print(f"\n Iteration: {i} \t Cost: {cost:.5f} \t Train Accuracy: {predict(X_train, Y_train, parameters, 'relu') * 100: .2f}% \t Test Accuracy: {predict(X_dev, Y_dev, parameters, activation) * 100: .2f}%")

    return parameters

def get_predictions(A2): 
    return np.argmax(A2, 0) ## this returns the index of the highest certainty

def make_predictions(parameters): 
    A2, _ = forward_prop(X_train, parameters, 'relu')
    pred = get_predictions(A2)
    return pred

def test_prediction(index, parameters): 
    current_image = X_train[:, index, None]
    AL, _ = forward_prop(current_image, parameters, 'relu')
    prediction = np.argmax(AL, axis=0)
    label = Y_train[index]
    
    print("Prediction:", np.squeeze(prediction))
    print("Actual:", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()

def save_params(parameters, filename="trained_params.pkl"): 
    with open(filename, 'wb') as f: 
        pickle.dump((parameters), f)
    print(f'Parameters saved to {filename}')

def load_saved_params(filename='trained_params.pkl'): 
    with open(filename, 'rb') as f: 
        parameters = pickle.load(f)
    print(f'Parameters loaded from {filename}')
    return parameters

def find_best_params(best_accuracy): 
    print("Finding best params for this model!")
    for rate in itertools.product(learning_rates):
        print(f'\nTraining with learning rate: {rate}')

        num_iterations = 25000 

        layers = [784, 128, 64, 10]


        parameters = model(X_train, Y_train, layers, learning_rate=rate, num_iterations=num_iterations) 

        accuracy = predict(X_dev, Y_dev, parameters, 'relu')

        print(f"\nFinal Test Accuracy: {accuracy * 100: .2f}")

        if accuracy > best_accuracy: 
            best_accuracy = accuracy
            best_hyperparams["learning rate"] = rate

if __name__ == "__main__": 

    ## Hyperparameters ## 
    learning_rate = 0.1
    num_iterations = 5000

    Y_dev = one_hot(Y_dev)
    Y_train = one_hot(Y_train)  

    layers = [784, 128, 64, 10]

    parameters = model(X_train, Y_train, layers, learning_rate=learning_rate, num_iterations=num_iterations)
    choice_to_save = str(input("Would you like to save these parameters? Y/N"))
    if choice_to_save == "Y" or choice_to_save == "y": 
        save_params(parameters)
    elif choice_to_save == "N" or choice_to_save == "n": 
        print("Parameters not saved!")
        sys.exit()

    # find_best_params(best_accuracy)
         
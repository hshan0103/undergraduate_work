#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:01:44 2020

@author: hshan
"""

import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from IPython.display import display 

import warnings
warnings. filterwarnings('ignore')

# Define model
class nn_:
    '''
    Input the sizes of hidden layers as list. 
    If building a single layer network, specified the first layer size, fst_layer. 
    Either argument, layers or fst_layer needed.
    
    '''
    def __init__(self, trainX, trainY, testX, testY, fst_layer=None, layers=[]):
        self.layers = layers
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.n = fst_layer
        if fst_layer == None and len(self.layers)==0: 
            raise Exception('Error hidden layers input, fst_layer or layers argument needed')
        
    def model(self, opt, epochs):
        
        model = Sequential()
        if len(self.layers)==0:
            model.add(Dense(self.n, input_dim = self.trainX.shape[1], activation = 'relu'))
        else:
            model.add(Dense(self.layers[0], input_dim = self.trainX.shape[1], activation = 'relu'))
            for i in range(1,len(self.layers)):
                model.add(Dense(i, activation = 'relu'))
        
       
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

        history = model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), epochs=epochs, verbose=0)

        # Evaluate the model
        _, train_eval = model.evaluate(self.trainX, self.trainY, verbose=0)
        _, test_eval = model.evaluate(self.testX, self.testY, verbose=0)
        print('Train: %.3f, Test: %.3f\n' % (train_eval, test_eval))
        return history, train_eval, test_eval

    # Plot history
    def visualize_adam_sgd(self, epochs, p_type=None, plot = False):
        '''
        Visualizing neural network with ADAM and SGD
        
        '''
        for e in epochs:
            if p_type == 'a':
                print(f'--- ADAM with epochs = {e} ---')
                history_adam, _, _ = self.model('adam', epochs = e)
                if plot==True:
                    plt.plot(history_adam.history['accuracy'], label = 'train_adam')
                    plt.plot(history_adam.history['val_accuracy'], label = 'test_adam')
                    plt.legend()
                    plt.show()
                    
                    plt.plot(history_adam.history['loss'], label = 'train_adam')
                    plt.plot(history_adam.history['val_loss'], label = 'test_adam')
                    plt.legend()
                    plt.show()
                    
            elif p_type == 's':
                print(f'--- SGD with epochs = {e} ---')
                history_sgd, _, _ = self.model('sgd', epochs = e)
                if plot==True:
                    plt.plot(history_sgd.history['accuracy'], label = 'train_sgd')
                    plt.plot(history_sgd.history['val_accuracy'], label = 'test_sgd')
                    plt.legend()
                    plt.show()

                    plt.plot(history_sgd.history['loss'], label = 'train_sgd')
                    plt.plot(history_sgd.history['val_loss'], label = 'test_sgd')
                    plt.legend()
                    plt.show()
            else:
                print(f'--- ADAM and SGD with epochs = {e} ---')
                history_adam, _, _ = self.model('adam', epochs = e)
                history_sgd, _, _ = self.model('sgd', epochs = e)
                if plot==True:
                    plt.plot(history_adam.history['accuracy'], label = 'train_adam')
                    plt.plot(history_adam.history['val_accuracy'], label = 'test_adam')
                    plt.plot(history_sgd.history['accuracy'], label = 'train_sgd')
                    plt.plot(history_sgd.history['val_accuracy'], label = 'test_sgd')
                    plt.legend()
                    plt.show()
                    
                    plt.plot(history_adam.history['loss'], label = 'train_adam')
                    plt.plot(history_adam.history['val_loss'], label = 'test_adam')
                    plt.plot(history_sgd.history['loss'], label = 'train_sgd')
                    plt.plot(history_sgd.history['val_loss'], label = 'test_sgd')
                    plt.legend()
                    plt.show()
        
    def visualize_(self, ep, learn_rate=[], momentum=[], plot = False):
        if len(learn_rate)>0:
        
            for l in learn_rate:
                opt = keras.optimizers.SGD(learning_rate = l)
                print(f'--- SGD with learning rate = {l} ---')
                history, _, _ = self.model(opt, epochs = ep)
            
                if plot==True:
                    plt.plot(history.history['accuracy'], label=f'train_{l}')
                    plt.plot(history.history['val_accuracy'], label=f'test_{l}')
                    plt.legend()
                    plt.show()
            
                    plt.plot(history.history['loss'], label=f'test_{l}')
                    plt.plot(history.history['val_loss'], label=f'test_{l}')
                    plt.legend()
                    plt.show()
        if len(momentum)>0:
            for m in momentum:
                opt = keras.optimizers.SGD(momentum = m)
                print(f'--- SGD with momentum = {m} ---')
                history, _, _ = self.model(opt, epochs=ep)
            
                if plot==True:
                    plt.plot(history.history['accuracy'], label=f'train_{m}')
                    plt.plot(history.history['val_accuracy'], label=f'test_{m}')
                    plt.legend()
                    plt.show()
            
                    plt.plot(history.history['loss'], label=f'test_{m}')
                    plt.plot(history.history['val_loss'], label=f'test_{m}')
                    plt.legend()
                    plt.show()


def parameter_effect(task, df, target, first, epochs_list, l_rate, momentums, layers_param, neurons):
    trainX, testX, trainY, testY = train_test_split(df, target, test_size = 0.3, random_state = 4661)
    if task in ['2', '3']:       
        #plot code
        #plot effect of different optimizer and visualized for a list of epochs 
        nn_single = nn_(trainX, trainY, testX, testY, fst_layer = first)
            
        if task =='2':
            nn_single.visualize_adam_sgd(epochs_list, plot = True)
        if task == '3':
            #fixed epochs to investigate the effect of learning rate and momentum
            nn_single.visualize_(3000, learn_rate=l_rate, plot= True)
            nn_single.visualize_(3000, momentum=momentums, plot= True)
                
    elif task == '4':    
        for layer in layers_param:
            nn_nlayers = nn_(trainX, trainY, testX, testY, layers = layer)
            print(f'--- {len(layer)} hidden layers: {layer} ---')
            nn_nlayers.visualize_adam_sgd([3000], p_type='s', plot =  True)
            #nn_nlayers.model('adam', 4000)
            
    elif task == '5': 
        for neuron in neurons:
            nn_neurons = nn_(trainX, trainY, testX, testY, fst_layer = neuron)
            print(f'--- single hidden layer with {neuron} neurons ---')
            nn_neurons.visualize_adam_sgd([3000], p_type='s', plot =  True)
            #nn_neurons.model('adam', 4000)
    else: 
        print('Invalid task entered.')

def main():

        df = pd.read_csv('/Users/hshan/Downloads/parkinson_data/processed_data.csv')
        target = df.Class
        df.drop('Class', axis = 1, inplace = True)
    
        N = 1
        epochs_list = [300, 1000, 3000]
        l_rate = [0.001, 0.01, 0.1]
        momentums = [0.001, 0.01, 0.1]
        first = 16
        layers_param = [[16,16], [16,16,16], [16,16,16,16]]
        neurons = [2,8,16,32]
        
        i = 0
        for i in range(N):
            i += 1
            print(f'--- Run: {i} --- \n')
            parameter_effect(task, df, target, first, epochs_list, l_rate, momentums, layers_param, neurons)


if __name__ == '__main__':
    while True:
        task = input('Enter the task number (2,3,4 or 5): ')
        main()
            

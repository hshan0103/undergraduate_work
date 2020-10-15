#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:13:52 2020

@author: hshan
"""

import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#pip install -U keras-tuner
import kerastuner as kt
from kerastuner.tuners import RandomSearch

import warnings
warnings. filterwarnings('ignore')


hp = kt.HyperParameters()
        
def nn_model(hp):
    model = Sequential()
    model.add(Dense(units = hp.Int('units', 12, 44, 4), input_dim = 26, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    lr = hp.Choice('learning_rate', [0.01, 0.005, 0.001])
    m = hp.Choice('momentum', [0.01, 0.005, 0.001])
    opt = keras.optimizers.SGD(lr, m)
    opt = hp.Choice('optimizer', ['adam', 'sgd'])
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


def tune(df, target, trial, epochs):
    trainX, testX, trainY, testY = train_test_split(df, target, shuffle = True, test_size = 0.3, random_state = 4661)  

    tuner = RandomSearch(nn_model, objective = 'val_accuracy', max_trials = trial, seed = 4661,
                         directory = 'my_dir', project_name = 'assessment')

    tuner.search(x = trainX, y = trainY, epochs = epochs, validation_data = (testX, testY), verbose = 0)   
    _, train_eval = tuner.get_best_models(num_models=1)[0].evaluate(trainX, trainY, verbose=0)
    _, test_eval = tuner.get_best_models(num_models=1)[0].evaluate(testX, testY, verbose=0)
    print('Train: %.3f, Test: %.3f\n' % (train_eval, test_eval))
    return tuner

def main():
    df = pd.read_csv('/Users/hshan/Downloads/parkinson_data/processed_data.csv')
    target = df.Class
    df.drop(['Class'], axis = 1, inplace = True)
    
    max_trial = 20
    tuned_epochs = 3000
    
    nn_tuned = tune(df, target, max_trial, tuned_epochs)
    
    return nn_tuned

if __name__ == '__main__':
    tuning = main()
    hyperparam = tuning.get_best_hyperparameters()[0]
    print('units:', hyperparam.get('units'))
    print('learning rate: ', hyperparam.get('learning_rate'))
    print('momentum', hyperparam.get('momentum'))
    
    
#trainX, testX, trainY, testY = train_test_split(df, target, shuffle = True, test_size = 0.3, random_state = 4661)
#model = Sequential()
#model.add(Dense(units = 44, input_dim = 26, activation = 'relu'))
#model.add(Dense(1, activation = 'sigmoid'))
#opt = keras.optimizers.Adam(0.01)
#model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
#z = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 1000, verbose = 0)
#plt.plot(z.history['accuracy'])
#plt.plot(z.history['val_accuracy'])
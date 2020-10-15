#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 09:53:02 2020

@author: hshan
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt

nn_dict = {'units': 12, 'l_rate':  0.005}
# SGD
# nn_dict = {'units': 44, 'l_rate':  0.01, 'momentum': 0.01}
def nn_best(nn_dict, df, target):
    model = Sequential()
    model.add(Dense(units = nn_dict['units'], activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    #opt = keras.optimizers.SGD(learning_rate = nn_dict['l_rate'], momentum = nn_dict['momentum'])
    opt = keras.optimizers.Adam(nn_dict['l_rate'])
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    trainX, testX, trainY, testY = train_test_split(df, target, shuffle = True, test_size = 0.3, random_state = 4661)
    history = model.fit(trainX, trainY, validation_data = (testX, testY), epochs=3000, verbose=0)
    
    class_hat = model.predict_classes(testX, verbose = 0)
    prob_hat = model.predict(testX, verbose = 0)
    fpr, tpr, threshold = roc_curve(testY, prob_hat)
    AUC = auc(fpr, tpr)

    print('Accuracy:', accuracy_score(testY, class_hat))
    plt.plot(history.history['val_accuracy'],label = 'test val_acc')
    plt.show()
    print(confusion_matrix(testY, class_hat))
    
    print('ROC-AUC Score:', AUC)
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()
    

    
df = pd.read_csv('/Users/hshan/Downloads/parkinson_data/processed_data.csv')
target = df.Class
df.drop(['Class'], axis = 1, inplace = True)

nn_best(nn_dict, df, target)

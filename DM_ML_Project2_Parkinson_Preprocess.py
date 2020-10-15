#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:33:40 2020

@author: hshan
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats 
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

class preprocessing:

    def check_na(data):
        df = data
        threshold = 0.01*df.shape[0]
        if df.isna().values.any():
            if df[df.isna().sum(axis=1)>0].shape[0] > threshold:
                print('Dataframe has more than 1% samples with missing values.')
                print(df.columns[df.isna().any()].tolist(), 'are features with missing values.')
                for col in df.columns[df.isna().any()].tolist():
                    df[col].fillna(df[col].mean())
            else:
                print('Dataframe has less than 1% samples with missing values.', 'The samples are dropped', sep = '/n')
                df.dropna(inplace = True)
        else:
            print('There is no missing value in dataframe.') 
        return df

    def outliers_transform(df):
        for col in df.columns.tolist():
            one, zero = boxplot_stats(df.loc[df.Class==1.0, col])[0], boxplot_stats(df.loc[df.Class==0.0, col])[0]
            if len(one['fliers'])!=0 or len(zero['fliers'])!=0:
                df[col] = np.sqrt(df[col])
        return df

    def outliers_capping(df):
        for col in df.columns.tolist():
            one, zero = boxplot_stats(df.loc[df.Class==1.0, col])[0], boxplot_stats(df.loc[df.Class==0.0, col])[0]
            if len(one['fliers'])!=0:
                min_1, max_1 = one['q1'] - 1.5*one['iqr'], one['q3'] + 1.5*one['iqr']
                df.loc[(df.Class==1.0) & (df[col]<min_1),col], df.loc[(df.Class==1.0) & (df[col]>max_1),col] = min_1, max_1
            if len(zero['fliers'])!=0:
                min_0, max_0 = zero['q1'] - 1.5*zero['iqr'], zero['q3'] + 1.5*zero['iqr']
                df.loc[(df.Class==0.0) & (df[col]<min_0),col], df.loc[(df.Class==0.0) & (df[col]>max_0),col] = min_0, max_0
        return df  


def main():
    col_names = ['Subject_id','Jitter_local', 'Jitter_local_absolute','Jitter_rap','Jitter_ppq5','Jitter_ddp', 
                 'Shimmer_local','Shimmer_local_dB','Shimmer_apq3','Shimmer_apq5', 'Shimmer_apq11','Shimmer_dda', 
                 'AC','NTH','HTN', 'Median_pitch','Mean_pitch','Standard_deviation','Minimum_pitch','Maximum_pitch', 
                 'Number_of_pulses','Number_of_periods','Mean_period','Standard_deviation_of_period', 
                 'Fraction_of_locally_unvoiced_frames','Number_of_voice_breaks','Degree_of_voice_breaks',
                 'UPDRS', 'Class']
    removed = ['Subject_id','UPDRS']

    df = np.loadtxt('/Users/hshan/Downloads/parkinson_data/raw_parkinson.txt', delimiter=',')
    df = pd.DataFrame(df, columns = col_names)

    df.drop(removed, axis=1, inplace = True)
    df.info()
    print('There are', df.shape[0], 'samples and', df.shape[1], 'numerical variables.')
    print('Target variable has label', df['Class'].unique().tolist())   
    # all are numerical variables, no categorical variables. 

    sns.countplot(x='Class', data=df)
    plt.show()
    # balanced datasets
    
    sns.pairplot(df, hue = 'Class')
    plt.show()
    
    sns.heatmap(df.corr(), xticklabels = df.columns.tolist(), yticklabels = df.columns.tolist())
    plt.show()
    # there are higly correlated features investigate the effect/impact

    df = preprocessing.check_na(df)

    for col in df.columns.tolist():
        if col != 'Class':
           sns.boxplot(x = 'Class', y = col, data = df)
           plt.show()
        
    # original data statistics
    print('-- statistics of original datasets --')
    display(df.describe())

    scaler = MinMaxScaler()
    df.iloc[:, :26] = scaler.fit_transform(df.iloc[:, :26])
    # scaled data statistics
    print('-- statistics of scaled datasets --')
    display(df.describe())

    df = preprocessing.outliers_transform(df)
    df = preprocessing.outliers_capping(df)
    
    # check nan for transformed data
    print('-- check presence of na for transformed and capped datasets --')
    display(df.isna().values.any())
    
    # transformed and capped data statistics
    print('-- statistics of transformed and capped datasets --')
    display(df.describe())
    
    df.to_csv('/Users/hshan/Downloads/parkinson_data/processed_data.csv', index = False)

if __name__ == '__main__':
    main()

    
    



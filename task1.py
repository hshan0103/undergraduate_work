# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:45:36 2020

@author: Hshan
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn import tree

from IPython.display import Image, display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings. filterwarnings('ignore')

class processing :
    '''
    processing data
    
    '''

    def main(df):
    
        df.info()
        print()
        print('-- Checking Null --')
    
        df = df.replace(r' ', np.NaN)
        if df.isnull().values.any():
            print('There is missing value in dataframe.')
            print(df.isnull().sum())
        else:
            print('There is no missing value in dataframe')
        
    def missing(df, col):
        '''
        drop the instances with missing value if the total rows with missing value is less than 1%
        ''' 

        df[col].replace(' ', np.nan, inplace = True)
        if df[col].isnull().sum()/df.shape[0] < 0.01:
            df.dropna(inplace = True)
        else:
            print(col, 'has more than 1% missing values, imputation needs to be considered')
        return df 
    
    def encoding(df, cat_cols, is_dummy = False):
        encoder = LabelEncoder()
        #scaler = StandardScaler()
        
        #df[num_cols] = scaler.fit_transform(df[num_cols])
        if is_dummy==False:
            for col in cat_cols:
                df[col] = encoder.fit_transform(df[col])
            return df
        else:
            df = pd.get_dummies(df, columns = cat_cols, drop_first=True)
            return df
    
    def string_to_float(df, cols):
        for col in cols:
            df[col] = df[col].apply(pd.to_numeric)
        return df

    def chisquare(df, cat_cols, target, threshold):
        '''
        label encoding categorical attributes and then conducting chi square test
        at significance level of 'threshold' 
        '''
        
        chi_df = df
        for col in cat_cols:
            chi_df[col] = LabelEncoder().fit_transform(df[col])
        score = chi2(chi_df[cat_cols], target)
        chi_selection = pd.DataFrame(np.transpose(score), index = cat_cols, columns = ['chi- square', 'p-value']).sort_values(by='p-value')
            
        feature_selected = chi_selection[chi_selection['p-value']<threshold].index
        display(chi_selection)
    
        return feature_selected
    
class train_eval:
    def __init__(self, df, target, test_size=0.2):
        self.df = df
        self.target = target
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.df, self.target, test_size= 0.2, random_state = 4661)
    
    def train(self, model, param):
        clf = GridSearchCV(model, param, n_jobs=-1, cv=7)
        clf.fit(self.xtrain, self.ytrain)
        
        plot_confusion_matrix(clf.best_estimator_, self.xtest, self.ytest)
        plt.show()
                                      
        print(classification_report(self.ytest, clf.best_estimator_.predict(self.xtest)))
        print(cv_results_)
        return clf.best_estimator_
        
    def plots(self, df_, estimator, is_dt=False):
        plt.figure(figsize=(10,10))
        plt.title('Feature Importances')
        plt.barh(list(df_.columns), estimator.feature_importances_, color='b', align='center')
        plt.xlabel('Features Importance')
        plt.tick_params(labelsize=15)
        plt.tight_layout()
        
        # tree visualization only available for decision tree
        if is_dt:
            dot = tree.export_graphviz(estimator, out_file=None, feature_names=list(df_.columns), class_names=['No', 'Yes'], filled=True)
            graph = pydotplus.graph_from_dot_data(dot)
            display(Image(graph.create_png()))

def with_percentage(plot, degree, x_, y_):
    for p in plot.patches:
        plot.annotate('{:.3f}%'.format(p.get_height()*100/7032), (p.get_x()+x_, p.get_height()+y_))
    plot.set_xticklabels(plot.get_xticklabels(), rotation=degree)
    plt.show()
    
def select_by_feature_importance(base, param, target, estimator_, df_, threshold):
    while np.any(estimator_.feature_importances_<=threshold) or np.any(np.isnan(estimator_.feature_importances_)):
        select = pd.DataFrame(np.transpose(estimator_.feature_importances_), index=df_.columns, columns = ['feature_importance'])
        select = select[select.feature_importance>threshold].sort_values(by='feature_importance')
        df_ = df_[select.index]
        training_ = train_eval(df_, target)

        estimator_ = training_.train(base, param)
    return df_, estimator_

    
dataframe = pd.read_csv('/Users/hshan/Downloads/data.csv')
    
target_cols = 'Churn'
    
id_cols = 'customerID'
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in list(dataframe .columns) if col not in num_cols]
    
removed = [id_cols] + [target_cols]
    
for element in removed:
    cat_cols.remove(element)

#df = processing.missing(data, 'TotalCharges')
#df = processing.string_to_float(df, ['TotalCharges'])
for c in ['TotalCharges', 'tenure']:
    q1_, q3_ = dataframe[dataframe.Churn=='Yes'][c].quantile(q=0.25),dataframe[dataframe.Churn=='Yes'][c].quantile(q=0.75)
    max_ = q3_+1.5*(q3_-q1_)
    dataframe.loc[(dataframe.Churn=='Yes') & (dataframe[c]>max_),c] = max_
    
target = dataframe[target_cols].eq('Yes').astype('int')
df = dataframe[dataframe.columns.difference(removed)]

#------------------------------------------------------------------------------------------------------------------
# chi-square test

chi_cat_cols = processing.chisquare(df, cat_cols, target, threshold=0.05)
chi_cols = list(chi_cat_cols) + num_cols
chi_df = processing.encoding(df[chi_cols], chi_cat_cols, is_dummy = True)

if chi_df.shape[0] != target.shape[0]:
    raise Exception('Input Error: Incompatible dimensions for x and y')

#------------------------------------------------------------------------------------------------------------------
# calling class train_eval with the appropriate argumenets

training = train_eval(chi_df, target)

#-----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # printing original loaded dataframe info
    processing.main(dataframe)
    print()
    
    # plot the target variable to inspect if its imbalanced 
    # and decide on the further processing steps
    with_percentage(sns.countplot(x=target_cols, data=dataframe), degree = 0, x_ = 0.25, y_ = 1)
    
    # visualized the numerical variables to check on distribution or possible correlation
    sns.pairplot(dataframe[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], plot_kws={'alpha':0.4}, hue = target_cols)
    sns.heatmap(dataframe[num_cols].corr(), xticklabels=num_cols, yticklabels=num_cols)
    
    # boxplot for identifying presence of outliers
    for col in num_cols: 
        sns.boxplot(x=col, y = target_cols, data=dataframe)
        plt.show()
        
    # plot barplots for categorical variables, observe and find interesting variables for reporting
    for col in cat_cols:
        with_percentage(sns.countplot(x=col, hue=target_cols, data=dataframe), degree = 45, x_ = 0.05, y_ = 1)

    # reporting the dataframe after omitting attributes under chi square testing
    print('Dimension of dataframe after chi-square testing for categorical variables:', chi_df.shape)

    # defining models and parameter lists
    dtree = DecisionTreeClassifier(criterion = 'gini')
    param = {'max_depth': range(1,20), 'min_samples_leaf': list(range(50, 80))}

    # training, evaluating and reporting decision tree
    estimator1 = training.train(dtree, param)
    df_dt, best_model_task1 = select_by_feature_importance(dtree, param, target, estimator1, chi_df, threshold=0.0)
    training.plots(df_dt, best_model_task1, is_dt=True)
    
    #DecisionTreeClassifier(max_depth=6, min_samples_leaf=65)
    
#------------------------------------------------------------------------------------------------------------------
# train with whole datasets without removing any feature

# print('-- Training decision tree with the original complete set of features --')
# training = train_eval(df, target)
# estimator = training.train(dtree, param)
# training.plots(estimator, is_dt = True)

# print('-- Training random forest with the original complete set of features --')
# estimator_rf = training.train(rf, param_rf)

# print('-- Training adaptive boosting with the original complete set of features --')
# estimator_ada = training.train(ada, param_ada)

#------------------------------------------------------------------------------------------------------------------
#from sklearn.inspection import permutation_importance
# result = permutation_importance(estimator, df, target, n_repeats=10, random_state=4661)
# result.importances_mean
# result.importances_std

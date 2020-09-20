#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:44:16 2020

@author: hshan
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from task1 import np, training, chi_df, target, select_by_feature_importance

if __name__ == '__main__':
    rf = RandomForestClassifier(criterion = 'gini', random_state = 4661)
    param_rf = {'n_estimators': range(10, 40),'min_samples_leaf': range(20, 40), 'max_depth':range(1,30,2)}

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), random_state = 4661)
    param_ada = {'learning_rate':np.linspace(0.1, 0.5,40), 'n_estimators': list(range(1,50))}

    estimator_rf1 = training.train(rf, param_rf)
    df_rf, estimator_rf = select_by_feature_importance(rf, param_rf, target, estimator_rf1, chi_df, threshold=0.0)
    training.plots(df_rf, estimator_rf, is_dt=False)
    print('best random forest model:', estimator_rf)
    
    estimator_ada1 = training.train(ada, param_ada)
    df_ada, estimator_ada = select_by_feature_importance(ada, param_ada, target, estimator_ada1, chi_df, threshold=0)
    training.plots(df_ada, estimator_ada, is_dt=False)
    print('best Adaboost classifier:', estimator_ada)

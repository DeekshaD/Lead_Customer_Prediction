import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

if __init__ == '__main__':
    
    data = pd.read_csv('loan_one_hot_encoded.csv')
    data = data[np.isfinite(data['amount'])]
    Y = data.loan_created
    #last_fy_profit for most unapproved loans is a 5-dig amt
    data.last_fy_profit[(Y==0) & (data.last_fy_profit.isnull())] = np.random.randint(10000,99999)

    #company size = 10 has the highest approved chances, while company size = 11 has lesser cchance
    data.loan_created[data.company_size==11].value_counts()

    data.company_size[(data.loan_created==1) & (data.company_size.isnull())]=10
    data.company_size[(data.loan_created==0) & (data.company_size.isnull())]=11

    data.age_of_firm[(data.loan_created==0) & (data.age_of_firm.isnull())]=np.average(data.age_of_firm[(data.loan_created==0)].dropna())
    data.age_of_firm[(data.loan_created==1) & (data.age_of_firm.isnull())]=np.average(data.age_of_firm[(data.loan_created==1)].dropna())

    data_new = data.drop(columns=['loan_created', 'application_id'])

    sm = SMOTE(random_state = 44, ratio = {0:210, 1:24}) #change class ratio accordingly

    X_os, Y_os = sm.fit_sample(data_new, Y)

    Y = pd.Series(Y_os)

    X = pd.DataFrame(X_os)

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    # parameters for GridSearchCV
    param_grid2 = {"n_estimators": [10, 22, 24, 20],
                   "max_features": ['auto', 'sqrt', 'log2'],
                  "max_depth": [8, 11, 27, None],
                  "min_samples_split": [2, 10, 12, 25],
                  "min_samples_leaf": [1, 2, 4, 14],
                  "max_leaf_nodes": [21, 13, 11, 9, None]}

    grid_search = GridSearchCV(clf, param_grid = param_grid2, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, Y)
    model_selection = pd.DataFrame(grid_search.cv_results_)
    
    print(grid_search.best_params_,"\n",grid_search.best_score_)

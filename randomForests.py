#author : Deeksha Doddahonnaiah, 1st year Northeastern University

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as met
from sklearn.metrics import roc_curve, roc_auc_score, auc, f1_score
from sklearn.model_selection import cross_val_score, train_test_split 
import warnings; warnings.simplefilter('ignore')
#%matplotlib inline

def scorer(clf, X, y, scoring='accuracy'):
    
    print('class: ',clf.class_weight)
    
    accuracy_ = cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy')
    print('Accuracy: ', np.average(accuracy_))
    
    f1_score_ = cross_val_score(clf, X, y, cv = 5, scoring = 'f1_macro')
    print('F1 score: ', np.average(f1_score_))
    
    auc_ =  cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc')
    print('AUC: ', np.average(auc_))
    
    print('\n\n')
    
    return accuracy_, f1_score_, auc_


def plot_auc(classifier, text,pt):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    plt.subplot(4,2,pt+1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for train, test in cv.split(syn_X, syn_Y):
    #     print(train,test)
        probas_ = classifier.fit(syn_X.iloc[train], syn_Y[train]).predict_proba(syn_X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(syn_Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = met.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = met.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic :\n '+text)
    plt.legend(loc="lower right", prop={'size':8})
    
def grid_search(X, Y):
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    clf = RandomForestClassifier(random_state = None, class_weight = {0:210, 1:9}) #change class weight appropriately
    # parameters for GridSearchCV
    param_grid2 = {"n_estimators": [10, 22, 24, 20],
                   "max_features": ['auto', 'sqrt', 'log2',None],
                  "max_depth": [8, 11, 27, None],
                  "min_samples_split": [2, 10, 12, 25],
                  "min_samples_leaf": [1, 2, 4, 14],
                  "max_leaf_nodes": [21, 13, 11, 9, None]}

    grid_search = GridSearchCV(clf, param_grid = param_grid2, cv=5, scoring=scoring, refit='AUC', n_jobs=-1)
    grid_search.fit(X, Y)
    model_selection = pd.DataFrame(grid_search.cv_results_)
    print(model_selection.columns)
    new_model = model_selection.sort_values(by='mean_test_Accuracy', inplace=False, ascending=False)

    new_model2 = model_selection.sort_values(by='mean_test_AUC', inplace=False, ascending=False)

    rank = new_model.index[:20]
    rank2 = new_model2.index[:20]



    m_count = 10
    for i in rank:
        print("Top 10 models with best Accuracy: \n")
        print("Model ",i,"\n",new_model.loc[i,'params'],"\nAuccuracy: ",new_model.loc[i,'mean_test_Accuracy'],"\nAUC: ", new_model.loc[i, 'mean_test_AUC'])
    print("\n\n")
    for i in rank2:
        print("Top 10 models with best AUC: \n")
        print("Model ",i,"\n",new_model2.loc[i,'params'],"\nAuccuracy: ",new_model2.loc[i,'mean_test_Accuracy'],"\nAUC: ", new_model2.loc[i, 'mean_test_AUC'])

if __name__ == '__main__':
    
    
    #original data
    org_data = pd.read_csv('data.csv')
    org_Y = org_data.loan_created
    org_X = org_data.drop(columns=['loan_created'])
    
    #synthesized data
    syn_data = pd.read_csv('syn_data.csv')
    syn_data = syn_data.drop(columns='Unnamed: 0')
    syn_Y = syn_data['loan_approved']
    syn_X = syn_data.drop(columns = ['loan_approved'])
    
    
    #The following has been done and the best parameters are chosen
    #Uncomment to see results of gridsearch
    
    # cw = {0:210,1:9}
    # grid_search(org_X, org_Y, cw)
    
    # cw = {-:210, 1:24}
    # grid_search(syn_X, syn_Y, cw)
    
    
    
    n_models = 4
    #model - 0 : Vanilla RF - default params
    #model - 1 : RF+GridSearch
    #model - 2 : Weighted RF
    #model - 3 : Balanced RF
    
    #Original data models:
    
    model_params_og = {}
    model_params_og[0] = {'random_state': 44}
    model_params_og[1] = {'max_depth': 8,
                    'max_features': 'log2', 
                    'max_leaf_nodes': 13,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 10,
                    'random_state': 44}
    model_params_og[2] = {'max_depth': 8,
                    'max_features': 'log2', 
                    'max_leaf_nodes': 13,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 10,
                    'random_state': 44,
                    'class_weight': {0:210, 1:9}}
    model_params_og[3] = {'max_depth': 8,
                    'max_features': 'log2', 
                    'max_leaf_nodes': 13,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 10,
                    'random_state': 44,
                    'class_weight': 'balanced'}


    #synthesized data params:
    
    model_params_syn = {}
    model_params_syn[0] = {'random_state': 87}
    model_params_syn[1] = {'max_depth': 27,
                    'max_features': 'sqrt', 
                    'max_leaf_nodes': None,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 20,
                    'random_state': 87}
    model_params_syn[2] = {'max_depth': 27,
                    'max_features': 'sqrt', 
                    'max_leaf_nodes': None,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 20,
                    'random_state': 87,
                    'class_weight': {0:210, 1:24}}
    model_params_syn[3] = {'max_depth': 27,
                    'max_features': 'sqrt', 
                    'max_leaf_nodes': None,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 20,
                    'random_state': 87,
                    'class_weight': 'balanced'}
    
    #Original data model runs:
    
    o_accuracy = {}
    o_f1 = {}
    o_auc = {}
    for i in range(n_models):
        print('ORIGINAL DATA')
        clf = RandomForestClassifier(**model_params_og[i])
        acc, f1, auc = scorer(clf, org_X, org_Y)
        o_accuracy[i] = acc
        o_f1[i] = f1
        o_auc[i] = auc
        
    # synthesized data model runs:
    print('*********************************************************************************************')
    
    s_accuracy = {}
    s_f1 = {}
    s_auc = {}
    for i in range(n_models):
        print('SMOTE DATA')

        clf = RandomForestClassifier(**model_params_syn[i])
        acc, f1, auc = scorer(clf, syn_X, syn_Y)
        s_accuracy[i] = acc
        s_f1[i] = f1
        s_auc[i] = auc
        
    #plots: uncomment if required:
    # n_samples, n_features = syn_X.shape
    # random_state = np.random.RandomState(0)
    # text = ['Default RF','GridSearch RF', 'GridSearch+Weighted RF', 'GridSearch+Balanced RF']

    # plt.figure(figsize=(12,18))
    # for i in range(4):
    #     cv = StratifiedKFold(n_splits=5)
    #     classifier = RandomForestClassifier(**model_params[i])
    #     plot_auc(classifier, text[i], i)
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 20:40:40 2018

@author: dhamnett
"""
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import interp
from itertools import product as cartesian_product



# fix random seed for reproducibility
SEED = 0
np.random.seed(SEED)

# Importing the datasets
dataset = pd.read_csv('Data/data_sameaxis.csv')
synthetic_dataset = pd.read_csv('Data/syn_data_sameaxis.csv')
full_dataset = pd.read_csv('Data/concat_data_sameaxis.csv')

# Getting the Data
X_data = dataset.iloc[:, :-1].values
y_data = dataset.iloc[:, -1].values
X_synth = synthetic_dataset.iloc[:, :-1].values
y_synth = synthetic_dataset.iloc[:, -1].values
X_full = full_dataset.iloc[:, :-1].values
y_full = full_dataset.iloc[:, -1].values

# Symbolic Constants
DROPOUT_RATE = .5
NUM_FEATURES = X_full.shape[1]
NUM_FOLDS = 10
ENSEMBLES = 10
BATCH_SIZE = 32
EPOCHS = 1000
BRIER = "Brier"
PRECISION = "Precision"
RECALL = "Recall"
F_SCORE = "F1"
ROC_AUC = "ROC-AUC"

# THE Input
INPUT = Input(shape=(NUM_FEATURES,))

# Single Hidden Layer Feed-Forward Neural Network
def get_network(hidden_layers=1):
    
    x = BatchNormalization()(INPUT)
    x = Dropout(DROPOUT_RATE)(x)
    for i in range(hidden_layers):
        x = Dense(NUM_FEATURES, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(INPUT, x)
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Ensemble of Classifiers
def get_ensemble(models):
    x = Average()([m.outputs[0] for m in models])
    model = Model(INPUT, x)
    return model

# Weighted Ensemble of Classifiers
def get_weighted_ensemble(X, models, quantity_to_weigh):
    prediction = np.zeros((X.shape[0],1))
    normalizer = sum(quantity_to_weigh)
    normalized_weights = [x / normalizer for x in quantity_to_weigh]
    for model, normalized_weight in zip(models, normalized_weights):
        prediction += model.predict(X)[:] * normalized_weight
    return prediction
    
# Keras Callback for ROC-AUC Score
# https://github.com/keras-team/keras/issues/3230#issuecomment-319208366
class roc_callback(Callback):
    
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
# Callbacks
def get_callbacks(training_data, validation_data):
    reduce_lr_loss = ReduceLROnPlateau()
    return [reduce_lr_loss, roc_callback(training_data, validation_data)]

# K-fold Wrapper
def get_folds(k=NUM_FOLDS, X=X_full, y=y_full, split=False):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    return folds.split(X, y) if split else folds

# Measureables
def get_metrics(y_test, y_pred):
    
    predicted_labels = np.rint(y_pred)
    brier_score = brier_score_loss(y_test, y_pred)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f_score = f1_score(y_test, predicted_labels)
    roc_auc_score = roc_auc_score(y_test, y_pred)
    return dict(zip([BRIER, PRECISION, RECALL, F_SCORE, ROC_AUC],[brier_score, precision, recall, f_score, roc_auc_score]))
    
# Extracting ROC-AUC from Neural Net
# def roc_auc(model, X_train, y_train, X_val, y_val):
#     y_pred = model.predict(X_train)
#     roc = roc_auc_score(y_train, y_pred)
#     y_pred_val = model.predict(X_val)
#     roc_val = roc_auc_score(y_val, y_pred_val)

    
# # Evaluation of Non-Ensemble Network
# estimator = KerasClassifier(build_fn=classification_model, epochs=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# results = cross_val_score(estimator, X_syn, y_syn, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


ensembles = []
# Run classifier with cross-validation and plot ROC curves
# Starter Code From:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def roc_auc_plt(X=X_full, y=y_full, model=get_network):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(get_folds(X=X, y=y, split=True)):
        classifier = model()
        classifier.fit(X[train], y[train], callbacks=get_callbacks((X[train], y[train]), (X[test], y[test])))
        ensembles.append(classifier)
        probas_ = classifier.predict(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    
# Prediction on entire Data Set
prediction = get_ensemble(models).predict(X_data)

# Plotting the Confusion Matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes=['Loan Not Approved','Loan Approved'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in cartesian_product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix([i for i in map(lambda x: 1 if x >= .5 else 0, prediction[:,0])], y_data)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,  
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
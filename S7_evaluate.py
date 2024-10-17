# -*- coding: utf-8 -*-
"""
Script for running the training and test phases of the S7a-g late fusion 
strategies for the classification of arrhythmia in the MIT-BIH dataset 
(https://www.kaggle.com/datasets/shayanfazeli/heartbeat/), as proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Sensors*, 2024.


Created on Thu Jul 11 19:30:21 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np


# Set main hyper-parameters
S = 'S7b'


# Set data folder
data_folder = './Data/'
save_folder = './Saved_Models/'
result_folder = './Results/'


# %% TRAINING

# Load training set
training_set = data_folder + 'S7_train_data.npy'
training_lab  = data_folder + 'S7_train_labels.npy'

X = np.load(training_set)
y = np.load(training_lab)


# Set the final classifier
from sklearn.svm import LinearSVC
clf_b = LinearSVC(C=1, random_state=42, tol=1e-4)

from sklearn.tree import DecisionTreeClassifier
clf_c = DecisionTreeClassifier(random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf_d = RandomForestClassifier(random_state=42, n_estimators=500, n_jobs=-1)

from sklearn.linear_model import LogisticRegression
clf_e = LogisticRegression(C=1, random_state=42, n_jobs=-1)

from sklearn.naive_bayes import GaussianNB
clf_f = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
clf_g = KNeighborsClassifier(n_neighbors=11)

from sklearn.linear_model import RidgeClassifier
clf_h = RidgeClassifier(alpha=1.0, solver='lsqr', random_state=42)


classifiers = {'S7b': clf_b, 'S7c': clf_c, 'S7d': clf_d, 'S7e': clf_e, 
               'S7f': clf_f, 'S7g': clf_g,}



model_name = S

if (S != 'S7a'):
    clf = classifiers[S]
    
    # Train the selected classifier
    clf.fit(X, y)



# %% TESTING

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


# Load test data
test_set = data_folder + 'S7_test_data.npy'
test_lab = data_folder + 'S7_test_labels.npy'

Xt = np.load(test_set)
yt = np.load(test_lab)


# Evaluate the model
if (S != 'S7a'):
    y_pred = clf.predict(Xt)
else:
    y_pred = np.argmax((Xt[:,:5] + Xt[:,5:])/2, axis=1)


# Evaluating the trained model
acc = accuracy_score(yt, y_pred)
pre = precision_score(yt, y_pred, average='weighted')
rec = recall_score(yt, y_pred, average='weighted')
f1  = f1_score(yt, y_pred, average='weighted')


# Printing metrics
print("Overall accuracy: {}%".format(round(100*acc,2)))
print("Precision: {}".format(round(pre,3)))
print("Recall: {}".format(round(rec,3)))
print("F1-score: {}".format(round(f1,3)))
# print("AUC: {}".format(round(AUC,3)))
print(" ", end='\n')
print("Complete report: ", end='\n')
print(classification_report(yt, y_pred))
print(" ", end='\n')


# Showing CM results 
labels = ['N', 'S', 'V', 'F', 'Q']  # 'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4

cm = confusion_matrix(yt, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
disp.plot(cmap='Blues')


# Save results on a text file
res_file = result_folder + 'Results_' + model_name + '.txt'
with open(res_file, 'a') as results:  # save the results in a .txt file
      results.write('-------------------------------------------------------\n')
      results.write('Acc: %s\n' % round(100*acc,2))
      results.write('Pre: %s\n' % round(pre,3))
      results.write('Rec: %s\n' % round(rec,3))
      results.write('F1: %s\n\n' % round(f1,3))
      results.write(classification_report(yt, y_pred, digits=3))
      results.write('\n\n')
      


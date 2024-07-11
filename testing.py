# -*- coding: utf-8 -*-
"""
This file contains main code for implementing the test phase for the
identification of arrhythmia in ECGs, proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Expert Systems With Applications*, 2024.


Created on Thu Jul 11 14:17:49 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import tensorflow as tf
import utils as ut

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report



# Select the strategy to use
S = 1


# Set data folder
data_folder = './Data/'
save_folder = './Saved_Models/'
result_folder = './Results/'


model_name = 'S' + str(S)
save_file = save_folder + model_name + '.h5'


# Load test set
Xt, yt, _ = ut.load_test_data(data_folder, S=S)


# Load the trained model
net = tf.keras.models.load_model(save_file)



# %% Evaluate the model output for test set
if (S<4):
    y_pred = net.predict(Xt)
else:
    y_pred = net.predict([Xt[0], Xt[1]])

y_pred = np.argmax(y_pred, axis=1)


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

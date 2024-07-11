# -*- coding: utf-8 -*-
"""
Script for obtaining the data to be used in the training and test phase of the 
S7 strategy for the late fusion of scalograms and phasograms of ECG in the 
MIT-BIH dataset (https://www.kaggle.com/datasets/shayanfazeli/heartbeat/), 
as proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Expert Systems With Applications*, 2024.


Created on Thu Jul 11 19:15:50 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import numpy as np

from tensorflow.keras.models import load_model


# Set data folder
data_folder = './Data/'
save_folder = './Saved_Models/'
result_folder = './Results/'



# %% Load the trained models

net_file_S = save_folder + 'S1.h5'
net_file_P = save_folder + 'S2.h5'

# Late fusion 
net_S = load_model(net_file_S)
net_P = load_model(net_file_P)


# %% TRAINING SET

# Load training set
training_set_S = data_folder + 'mitbih_train_scalograms.npy'
training_set_P = data_folder + 'mitbih_train_phasograms.npy'
training_lab   = data_folder + 'mitbih_train_labels.npy'

X_s = np.load(training_set_S)
X_p = np.load(training_set_P)
y  = np.load(training_lab)


# Shuffle the dataset
np.random.seed(seed=42)
idx = np.random.permutation(len(y))
X_s = X_s[idx,:,:]
X_p = X_p[idx,:,:]
y   = y[idx]


# Evaluate the single outputs and concatenate them
y_s = net_S.predict(X_s)
y_p = net_P.predict(X_p)

y_o = np.concatenate([y_s, y_p], axis=1)


# Saving files
save_file_1 = data_folder + 'S7_train_data.npy'
save_file_2 = data_folder + 'S7_train_labels.npy'

np.save(save_file_1, y_o)
np.save(save_file_2, y)



# %% TEST SET

# Load test set
test_set_S = data_folder + 'mitbih_test_scalograms.npy'
test_set_P = data_folder + 'mitbih_test_phasograms.npy'
test_lab   = data_folder + 'mitbih_test_labels.npy'

Xt_s = np.load(test_set_S)
Xt_p = np.load(test_set_P)
yt   = np.load(test_lab)


# Evaluate the model output for test set
y_pred_m = net_S.predict(Xt_s)
y_pred_p = net_P.predict(Xt_p)

y_pred_o = np.concatenate([y_pred_m, y_pred_p], axis=1)


# Saving files
save_file_3 = data_folder + 'S7_test_data.npy'
save_file_4 = data_folder + 'S7_test_labels.npy'

np.save(save_file_3, y_pred_o)
np.save(save_file_4, yt)

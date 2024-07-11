# -*- coding: utf-8 -*-
"""
Script for extracting the magnitude (scalogram) and phase (phasogram) of the
Continuous Wavelet Transform (CWT) of ECG signals in the MIT-BIH dataset
(https://www.kaggle.com/datasets/shayanfazeli/heartbeat/), to be used in the
work proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Expert Systems With Applications*, 2024.


Created on Mon Nov 27 21:52:26 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import os
import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt
from ssqueezepy import cwt


# Scale matrix in range [0, 1] ------------------------------------------------
def scale(matrix):
    # Perform min-max scaling
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    scaled_matrix = (matrix - min_val) / (max_val - min_val)

    return scaled_matrix
#------------------------------------------------------------------------------



# Set folder and hyper-parameters
data_folder = './Data/'

# CWT parameters
fs = 125   # Sample rate
N  = 224   # Number of pixels

# TRaining and test sets
sets = ['train', 'test']


# Loop over sets
for set in sets:
    file_name = data_folder + 'mitbih_' + set + '.csv'

    # Read the dataset and extract features and labels
    data = pd.read_csv(file_name, header=None)

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y = y.astype(int)

    # B = np.bincount(y)

    L = X.shape[0]

    feat_Wm = []
    feat_Wp = []


    # Main loop
    for i in range(L):
        # Extract the CWT
        W, scales = cwt(X[i,:], wavelet='morlet', fs=fs)

        # Compute the scalogram and its phasogram
        Wm = np.abs(W)
        Wp = np.angle(W)

        # Resize to a suitable image size (e.g., 224x224 or 227x227)
        Wm = cv2.resize(Wm, dsize=(N, N), interpolation=cv2.INTER_LINEAR)
        Wp = cv2.resize(Wp, dsize=(N, N), interpolation=cv2.INTER_LINEAR)

        # Normalize features
        Wm = scale(Wm)
        Wp = scale(Wp)

        # Transform as an integer image
        Wm = np.array(255*Wm, dtype = 'uint8')
        Wp = np.array(255*Wp, dtype = 'uint8')

        # Append features
        feat_Wm.append(Wm)
        feat_Wp.append(Wp)

        # Print advancement
        if (i % 100):
            print("\rAdvancement: {}%".format(round(100*i/L, 1)), end='')

    print("\rAdvancement: {}%".format(100.0), end='\n')


    # Save the extracted scalograms, phasograms, and labels
    save_file_m = file_name[:-4] + '_scalograms.npy'
    save_file_p = file_name[:-4] + '_phasograms.npy'
    save_file_l = file_name[:-4] + '_labels.npy'
    np.save(save_file_m, feat_Wm)
    np.save(save_file_p, feat_Wp)
    np.save(save_file_l, y)
    print('Done for set:', set)

print('Done!')

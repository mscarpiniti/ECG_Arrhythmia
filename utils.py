# -*- coding: utf-8 -*-
"""
This file contains useful function used in the main code implementing the
identification of arrhythmia in ECGs, proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Expert Systems With Applications*, 2024.


Created on Wed Jul 10 18:09:29 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import models
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical




# Function for loading the training data to be used in the proposed model -----
def load_training_data(data_folder, NB=32, S=1):
    """
    Function for loading the training data for the proposed model.


    Parameters
    ----------
    data_folder : folder containing data.
    NB : integer equal to the batch size.
    S : integer equal to the strategy to be used. The default is 1.

    Returns
    -------
    dataset : a TensorFlow Dataset object.

    """

    # Load the data
    if (S == 1):
        training_set = data_folder + 'mitbih_train_scalograms.npy'

        X = np.load(training_set)
        L = X.shape[0]
    elif (S==2):
        training_set = data_folder + 'mitbih_train_phasograms.npy'

        X = np.load(training_set)
        L = X.shape[0]
    elif (S==3):
        training_set_s = data_folder + 'mitbih_train_scalograms.npy'
        training_set_p = data_folder + 'mitbih_train_phasograms.npy'

        X_s = np.load(training_set_s)
        X_p = np.load(training_set_p)

        X_s = X_s[:,:,:,np.newaxis]
        X_p = X_p[:,:,:,np.newaxis]
        X = np.concatenate((X_s, X_p), axis=3)
        L = X.shape[0]
    elif ((S>=4) and (S<7)):
        training_set_s = data_folder + 'mitbih_train_scalograms.npy'
        training_set_p = data_folder + 'mitbih_train_phasograms.npy'

        X_s = np.load(training_set_s)
        X_p = np.load(training_set_p)
        L = X_s.shape[0]
    else:
        X = []
        print("The strategy number must not exeed 6!")


    # Set the permutation vector
    np.random.seed(seed=42)
    idx = np.random.permutation(L)

    # Load the labels
    training_lab = data_folder + 'mitbih_train_labels.npy'
    y = np.load(training_lab)
    y = y[idx]
    y_cat = to_categorical(y, 5) # Convert to categorical value

    # Shuffle the data and create the dataset object
    if (S<3):
        X = X[idx,:,:]
        dataset = tf.data.Dataset.from_tensor_slices((X, y_cat))
    elif (S==3):
        X = X[idx,:,:,:]
        dataset = tf.data.Dataset.from_tensor_slices((X, y_cat))
    else:
        X_s = X_s[idx,:,:]
        X_p = X_p[idx,:,:]
        dataset = tf.data.Dataset.from_tensor_slices(((X_s,X_p), y_cat))

    dataset = dataset.batch(NB)


    return dataset






# Function for loading the test data to evaluate the proposed model -----------
def load_test_data(data_folder, S=1):
    """
    Function for loading the test data for validating the proposed model.


    Parameters
    ----------
    data_folder : folder containing data.
    S : integer equal to the strategy to be used. The default is 1.

    Returns
    -------
    Xt : test feature matrix.
    yt : test labels.
    yt_cat : test labels in categorical values.

    """

    # Load the data
    if (S==1):
        test_set = data_folder + 'mitbih_test_scalograms.npy'
        Xt = np.load(test_set)
    elif (S==2):
        test_set = data_folder + 'mitbih_test_phasograms.npy'
        Xt = np.load(test_set)
    elif (S==3):
        test_set_s = data_folder + 'mitbih_test_scalograms.npy'
        test_set_p = data_folder + 'mitbih_test_phasograms.npy'

        Xt_s = np.load(test_set_s)
        Xt_p = np.load(test_set_p)

        Xt_s = Xt_s[:,:,:,np.newaxis]
        Xt_p = Xt_p[:,:,:,np.newaxis]
        Xt = np.concatenate((Xt_s, Xt_p), axis=3)
    elif ((S>=4) and (S<7)):
        test_set_s = data_folder + 'mitbih_test_scalograms.npy'
        test_set_p = data_folder + 'mitbih_test_phasograms.npy'

        Xt_s = np.load(test_set_s)
        Xt_p = np.load(test_set_p)
        Xt = (Xt_s, Xt_p)
    else:
        Xt = []
        print("The strategy number must not exeed 6!")


    # Load the labels
    test_lab = data_folder + 'mitbih_test_labels.npy'
    yt = np.load(test_lab)

    # Convert to categorical labels
    yt_cat = to_categorical(yt, 5)


    return Xt, yt, yt_cat




# Function for loading the selected mode to be used in the proposed strategy --
def select_model(S=1, LR=0.001):
    """
    Function for selecting the model to be trained.


    Parameters
    ----------
    S : integer equal to the strategy to be used. The default is 1.
    LR: (float) the learning rate

    Returns
    -------
    net : the TensorFlow network model.

    """

    # Selecting the model
    if (S==1):
        net = models.S1(LR)
    elif (S==2):
        net = models.S2(LR)
    elif (S==3):
        net = models.S3(LR)
    elif (S==4):
        net = models.S4(LR)
    elif (S==5):
        net = models.S5(LR)
    elif (S==6):
        net = models.S6(LR)
    else:
        net = []
        print("The strategy number must not exeed 6!")


    return net

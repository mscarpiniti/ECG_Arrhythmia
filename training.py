# -*- coding: utf-8 -*-
"""
This file contains main code for implementing the training phase for the
identification of arrhythmia in ECGs, proposed in [1].

[1] M. Scarpiniti, "Arrhythmia detection by data fusion of ECG scalograms and
phasograms", submitted to *Expert Systems With Applications*, 2024.


Created on Wed Jul 10 19:41:14 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import numpy as np
import tensorflow as tf
import models
import utils as ut
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping



# Select the strategy to use
S = 1


# Set main hyper-parameters
LR = 0.0001  # Learning rate
N_b = 32     # Batch size
N_e = 30     # Number of epochs
Pat = 5      # Patience for early-stopping


# Set data folder
data_folder = './Data/'
save_folder = './Saved_Models/'
result_folder = './Results/'



# Load training set
with tf.device('CPU/:0'):
    dataset = ut.load_training_data(data_folder, NB=N_b, S=S)




# %% Select and train the model
model_name = 'S' + str(S)

net = ut.select_model(S, LR)


# Early stopping
early_stop = EarlyStopping(monitor='accuracy', patience=Pat, restore_best_weights=True)


# Train the selected model
history = net.fit(dataset, epochs=N_e, shuffle=True, callbacks=[early_stop])


# Save the trained model and history
save_file = save_folder + model_name + '.h5'
net.save(save_file, overwrite=True, include_optimizer=True, save_format='h5')
np.save(save_folder + model_name + '_history.npy', history.history)



# %% Plot curves
# history = np.load(save_folder + model_name + '_history.npy', allow_pickle='TRUE').item()
L_e = len(history.history['loss'])
ep = range(1, L_e+1)

# Plot loss curve
plt.figure()
plt.plot(ep, history.history['loss'], linewidth=2, label='Training loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
# fig_name = save_folder + model_name + '_Loss.pdf'
# plt.savefig(fig_name, format='pdf')


# Plot accuracy curve
plt.figure()
plt.plot(ep, history.history['accuracy'], linewidth=2, label='Training accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
# fig_name = save_folder + model_name + '_Accuracy.pdf'
# plt.savefig(fig_name, format='pdf')

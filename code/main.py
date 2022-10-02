############################################################
# Load tmp_data.npz as data and train specified model.     #
# classification_report will save to 'report.csv'          #
#                                                          #
# Usage: main.py <model id> <epochs> <batch_size>         #
############################################################

import sys
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

from model import ANN_model, CNN_model, DNN_model, MLP_model, LSTM_model
from train import train_model

if len(sys.argv) != 4:
    print('-> Usage: main.py <model id> <epochs> <batch_size>')
    sys.exit()

# Load data generated from process_dataset.py
tmp_data = np.load('tmp_data.npz')
X_train = tmp_data['X_train']
X_test = tmp_data['X_test']
Y_train = tmp_data['Y_train']
Y_test = tmp_data['Y_test']
input_size = X_train.shape[1]
label_count = len(np.unique(Y_train))

# Load model
models = [ANN_model, CNN_model,DNN_model, MLP_model, LSTM_model]
model_generator = models[int(sys.argv[1])]
model = model_generator(input_size, label_count)

# Train model
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
precision, recall, f1score, Training_time, report = train_model(model, X_train, X_test, Y_train, Y_test, epochs = epochs, batch_size = batch_size)

# Saving report
df_report = pd.DataFrame(report)
df_report["Total_training_time(s)"]=[Training_time, '', '', '']
df_report["Input_size"]=[input_size, '', '', '']
df_report = df_report.transpose()
df_report.to_csv('report.csv')

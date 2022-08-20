############################################################
# All model made by us.                                    #
#                                                          #
# Sample usage:                                            #
#     from model import ANN_model                          #
#     mymodel = ANN_model(86)                              #
############################################################

from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, AveragePooling1D, MaxPooling1D, Flatten
import tensorflow as tf

def ANN_model(input_size):
    print('--> ANN_model')
    model = Sequential()
    InputLayer = Input( batch_input_shape=(None, input_size ), name="dense_input", dtype=tf.float32, sparse=False, ragged=False)
    model.add(InputLayer)
    model.add(Dense(256, batch_input_shape=(None, input_size ), name="dense", dtype=tf.float32, activation="relu"))
    model.add(Dense(128, name="dense_1", dtype=tf.float32, activation="relu"))
    model.add(Dense(39, name="dense_2", dtype=tf.float32, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

def CNN_model(input_size):
    print('--> CNN_model')
    model = Sequential()
    InputLayer = Input( batch_input_shape=(None, input_size, 1), name="conv1d_input", dtype=tf.float32, sparse=False, ragged=False)
    model.add(InputLayer)
    model.add(Conv1D(filters=10, batch_input_shape=(None, input_size, 1), name="conv1d", kernel_size=2, dtype=tf.float32, activation="relu"))
    model.add(Conv1D(filters=10, name="conv1d_1", kernel_size=2, dtype=tf.float32, activation="relu"))
    model.add(AveragePooling1D(name="average_pooling1d", strides=2, dtype=tf.float32))
    model.add(MaxPooling1D(name="max_pooling1d", pool_size=3, strides=3, dtype=tf.float32))
    model.add(Flatten(name="flatten", dtype=tf.float32))
    model.add(Dense(39, name="dense", dtype=tf.float32, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

def DNN_model(input_size):
    print('--> DNN_model')
    model = Sequential()
    InputLayer = Input( batch_input_shape=(None, input_size ), name="dense_input", dtype=tf.float32, sparse=False, ragged=False)
    model.add(InputLayer)
    model.add(Dense(512, batch_input_shape=(None, input_size ), name="dense", dtype=tf.float32, activation="relu"))
    model.add(Dense(256, name="dense_1", dtype=tf.float32, activation="relu"))
    model.add(Dense(128, name="dense_2", dtype=tf.float32, activation="relu"))
    model.add(Dense(39, name="dense_3", dtype=tf.float32, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

def MLP_model(input_size):
    print('--> MLP_model')
    model = Sequential()
    InputLayer = Input( batch_input_shape=(None, input_size), name="dense_input", dtype=tf.float32, sparse=False, ragged=False)
    model.add(InputLayer)
    model.add(Dense(128, batch_input_shape=(None, input_size), name="dense", dtype=tf.float32, activation="relu"))
    model.add(Dense(39, name="dense_2", dtype=tf.float32, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

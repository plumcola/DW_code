import boto
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from plot_metrics import plot_accuracy, plot_loss, plot_roc_curve
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,Bidirectional
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D,LSTM,Reshape
from keras.utils import np_utils
from keras import backend as K
from tensorflow.keras.losses import MSE
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os
from torchvision import transforms



def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))
    print('Train on {} samples, validate on {}'.format(y_train.shape[0],
                                                       y_test.shape[0]))

    # normalize to dBfS
    X_train, X_test = preprocess(X_train, X_test)


    return X_train, X_test, y_train, y_test


def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)


    return X_train, X_test, input_shape


def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs, input_shape):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1,
                     input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
              input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Reshape((32, -1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                        optimizer=RMSprop(lr=0.001),
                        metrics=['acc'])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=2, validation_data=(X_test, y_test),shuffle=True)

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history



def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    yu=0.45
    y_test_1=[]
    y_train_1=[]
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    for i in y_test_pred:
        if i>=yu:
            y_test_1.append(1)
        else:
            y_test_1.append(0)
    for i in y_train_pred:
        if i>=yu:
            y_train_1.append(1)
        else:
            y_train_1.append(0)
    y_train_1=np.array(y_train_1)
    y_test_1 = np.array(y_test_1)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test
    y_train_1d = y_train
    conf_matrix=standard_confusion_matrix(y_test_1d, y_test_1)
    #conf_matrix = confusion_matrix(y_test_1d, y_test_1)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_1, y_test_1, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])
    #return [[tn, fp], [fn, tp]]

def dataIter(batch_size,trainX, trainY):

    trainX1 = torch.tensor(trainX)  # 这里必须是tensor
    trainY1 = torch.tensor(trainY,dtype=torch.long)
    dataset = TensorDataset(trainX1, trainY1)
    train_data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return trainX, trainY,train_data_iter


if __name__ == '__main__':
    model_id = "processed_mfcc_2"

    X = np.load("./"+model_id+"/train_samples.npz",allow_pickle=True)
    y = np.load("./"+model_id+"/train_labels.npz",allow_pickle=True)
    X_test = np.load("./"+model_id+"/test_samples.npz",allow_pickle=True)
    y_test = np.load("./"+model_id+"/test_labels.npz",allow_pickle=True)

    X, y = X['arr_0'], y['arr_0']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


    # CNN parameters
    batch_size = 128
    nb_classes = 1
    epochs = 7

    #normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train, X_test, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)

    # 513x125x1 for spectrogram with crop size of 125 pixels
    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1
    print(img_rows, img_cols, img_depth)
    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    X_train, X_test, input_shape = keras_img_prep(X_train, X_test, img_depth,
                                                  img_rows, img_cols)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    #run CNN
    print('Fitting model...')


    model, history = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs, input_shape)

    




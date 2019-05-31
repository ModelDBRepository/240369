import tensorflow as tf
import numpy as np
from scipy import misc
import keras
import glob
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
from keras import regularizers
import os
import sys
from sys import argv
from keras.callbacks import CSVLogger


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

__input_file__ = "caltech/" #Address to dataset folder

myargs = getopts(argv)

__output_file__ = myargs['-o']
__gpuid__ = "/gpu:" + myargs['-g']


with tf.device(__gpuid__):
    objects = ["faces" , "motors"]
    mapping = {"faces" : 0 , "motors" : 1}
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_val = []
    Y_val = []

    if not os.path.exists(__output_file__):
        os.makedirs(__output_file__)

    
    for name in objects:
        #importing the training dataset
        perm = np.random.permutation(range(1, 436))
        train = perm[0:200]
        test = perm[200:435]
        postfix = ".png"
        space = ""
        if(name == 'motors'):
            space = " "
        for i in train:
            image_path = __input_file__ + name + "/" + space + str(i) + postfix
            image = misc.imread(image_path)
            image = rgb2gray(image)
            X_train.append(image)
            Y_train.append(mapping[name])
        for i in test:
            image_path = __input_file__ + name + "/" + space + str(i) + postfix
            image = misc.imread(image_path)
            image = rgb2gray(image)
            X_val.append(image)
            Y_val.append(mapping[name])

    x_train = np.asarray(X_train)
    y_train = np.asarray(Y_train)
    x_val = np.asarray(X_val)
    y_val = np.asarray(Y_val)
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    x_train /= (np.max(x_train) - np.min(x_train))
    x_val /= (np.max(x_val) - np.min(x_val))
    
    x_train = x_train.reshape(x_train.shape[0] , 150, 227 ,1 )
    x_val = x_val.reshape(x_val.shape[0] , 150 , 227 ,1 )
    
    Y_train = np_utils.to_categorical(y_train, 2)
    Y_val = np_utils.to_categorical(y_val , 2)

    model = Sequential()
    model.add(Convolution2D(4, (5, 5), activation='relu', kernel_regularizer = regularizers.l2(0.001) ,input_shape=(150,227,1) ))
    model.add(MaxPooling2D(pool_size=(7,7) , strides = (6,6)))
    model.add(Convolution2D(20 , (24,37) , activation='relu' , kernel_regularizer = regularizers.l2(0.001) ))
    
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(70 , activation='relu' ,kernel_regularizer = regularizers.l2(0.005) ))
    model.add(Dropout(0.5))
    model.add(Dense(2 , activation= 'softmax' , kernel_regularizer = regularizers.l2(0.03) , activity_regularizer = regularizers.l1(1) ))

    csv_logger = CSVLogger(__output_file__ + "/result.csv")

    opt = keras.optimizers.SGD(lr=0.01, momentum=0, decay=0.0002, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    
    history = model.fit(x_train, Y_train, shuffle = True ,
            batch_size=16, validation_data = (x_val , Y_val) ,epochs = 700 , callbacks = [csv_logger] , verbose=1 )
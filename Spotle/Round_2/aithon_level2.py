#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,Model 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Input,BatchNormalization 
from keras.optimizers import RMSprop,Adam


'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''


def aithon_level2_api(trainingcsv, testcsv):

    # Training the model with training data
    train = pd.read_csv(trainingcsv)
    
    Y_train = train['emotion']
    Y_train = Y_train.to_numpy()
    
    n = Y_train.shape[0]
    X_train = train.drop('emotion',axis = 1)
    X_train = X_train.to_numpy()
    X_train = X_train/255.0
    X_train = X_train.reshape([n,48,48,1])
    
    Y_train_num = []
    for i in range(n):
        if(Y_train[i] == 'Happy'):
            Y_train_num.append(0)
        elif (Y_train[i] == 'Sad'):
            Y_train_num.append(1)
        else:
            Y_train_num.append(2)
            
    Y_onehot_train = to_categorical(Y_train_num,num_classes=3) #Converts Label into one-hot Vectors
    
    '''
    A Keras' Convolution Model with 6 Conv Layers and 3 MaxPooling Layers. Output of the Model is made using a 
    Softmax Layer and it gives a 1X3 vector.
    '''
    
    Model = Sequential([
        
        Conv2D(64,(5,5),activation = 'relu',padding = 'same',input_shape=(48,48,1)),
        Conv2D(64,(5,5),activation = 'relu',padding = 'same'),
        BatchNormalization(),
        MaxPool2D(2,2),
        Conv2D(128,(5,5),activation = 'relu',padding = 'same'),
        Conv2D(128,(5,5),activation = 'relu',padding = 'same'),
        BatchNormalization(),
        MaxPool2D(2,2),
        Conv2D(256,(5,5),activation = 'relu',padding = 'same'),
        Conv2D(256,(5,5),activation = 'relu',padding = 'same'),
        BatchNormalization(),
        MaxPool2D(2,2),
        Flatten(),
        Dense(128,activation = 'relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(3,activation = 'softmax')
    ])
    
    Model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    history = Model.fit(X_train,Y_onehot_train,epochs = 25,batch_size = 1000,verbose = 1)
    
    # Testing the Model using Test Data
    
    test = pd.read_csv(testcsv)
    test = test.to_numpy()
    test = test.reshape([-1,48,48,1])
    test = test/255.0
    m = test.shape[0]
    
    temp = Model.predict(test)
    prediction = []
    
    for i in range(m):
        value = np.argmax(temp[i])
        if(value == 0):
            prediction.append('Happy')
        elif(value == 1):
            prediction.append('Sad')
        else:
            prediction.append('Fear')
            
            
    return prediction # A List containing Predicitons for Test_Data

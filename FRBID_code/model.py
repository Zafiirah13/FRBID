#!usr/bin/env python
"""
Authors : Zafiirah Hosenie
Email : zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk
Affiliation : The University of Manchester, UK.
License : MIT
Status : Under Development
Description :
Python implementation for FRBID: Fast Radio Burst Intelligent Distinguisher.
This code is tested in Python 3 version 3.5.3  
"""

import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator



def get_modelparameters(params,img_shape, lr):
    '''
    This function calls out the model we want for training the images

    INPUT
        params: The model name we want to train for e.g 'NET1_32_64', 'NET1_64_128', 'NET1_128_256', 'NET2', 'NET3'
        img_shape: The shape of the image (256,256,2), or (256, 256, 1), or (X_train.shape[1],X_train.shape[2],X_train.shape[3])
        lr: The learning rate for the optimisation values can vary from [0.1, 0.01, 0.001, 0.0001]
    '''

    if params == 'NET1_32_64':

        A = 32; B = 64

        model = Sequential()
        model.add(Conv2D(A, (3, 3),activation='relu', input_shape=img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Dense(B))
        model.add(Dropout(0.5))
        model.add(Dense(B))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.summary()
        optimizers = keras.optimizers.Adam(lr=lr)
        losses = 'binary_crossentropy'
        model.compile(optimizer=optimizers, loss=losses, metrics=['accuracy'])
        
        return model

    elif params == 'NET1_64_128':

        A = 64; B = 128

        model = Sequential()
        model.add(Conv2D(A, (3, 3),activation='relu', input_shape=img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Dense(B))
        model.add(Dropout(0.5))
        model.add(Dense(B))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.summary()
        optimizers = keras.optimizers.Adam(lr=lr)
        losses = 'binary_crossentropy'
        model.compile(optimizer=optimizers, loss=losses, metrics=['accuracy'])
        
        return model

    elif params == 'NET1_128_256':

        A = 128; B = 256

        model = Sequential()
        model.add(Conv2D(A, (3, 3),activation='relu', input_shape=img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Dense(B))
        model.add(Dropout(0.5))
        model.add(Dense(B))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.summary()
        adam_op = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=adam_op, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    elif params == 'NET2':

        A = 32; B = 128; C = 512


        model = Sequential()
        model.add(Conv2D(A, (3, 3),activation='relu', input_shape=img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(B, (3, 3),activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Dense(C))
        model.add(Dropout(0.5))
        model.add(Dense(C))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.summary()
        adam_op = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=adam_op, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    elif params == 'NET3':

        A = 16; B = 32; C = 64; D = 1000

        model = Sequential()
        model.add(Conv2D(A, (3, 3),activation='relu', input_shape=img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(B, (3, 3),activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(C, (3, 3),activation='relu')),
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Dense(D))
        model.add(Dropout(0.5))
        model.add(Dense(D))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.summary()
        adam_op = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=adam_op, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model


def compile_model(params,img_shape,save_model_dir, X_train, y_train, X_val, yval1h, batch_size, epochs, lr, class_weight, early_stopping=False,save_model=False,data_augmentation=False):
    '''
    This function compile the model, apply early stopping to avoid overfitting and also apply data augmentation
    if we set it to True

    INPUTS:
        params: The model name we want to train for e.g 'NET1_32_64', 'NET1_64_128', 'NET1_128_256', 'NET2', 'NET3'
        img_shape: The shape of the image (256,256,2), or (256, 256, 1), or (X_train.shape[1],X_train.shape[2],X_train.shape[3])
        save_model_dir: The directory we want to save the history of the model [accuracy, loss]
        X_train, X_val: The training set and validation sey having shape (Nimages, 256pix, 256pix, 2 images)
        y_train, yval1h: The label for training and validation set- transform to one-hot encoding having shape (Nimages, 2) in the format array([[0., 1.],[1., 0.])
        batch_size: Integer values values can be in the range [32, 64, 128, 256]
        epoch: The number of iteration to train the network. Integer value varies in the range [10, 50, 100, 200, ...]
        lr: The learning rate for the optimisation values can vary from [0.1, 0.01, 0.001, 0.0001]
        class_weight: If we want the model to give more weights to the class we are interested then set it to {0:0.25,1:0.75} or None
        early_stopping: Stop the network from training if val_loss stop decreasing if TRUE
        save_model: set TRUE to save the model after training
        data_augmentation: set TRUE if we want to apply data augmentation

    OUTPUTS:
        history: The logs of the accuracy and loss during optimization
        modelCNN: The fully trained model

    '''
    if save_model:
        tensorboard = TensorBoard(log_dir = save_model_dir + 'logs', write_graph=True)
 
    callbacks = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
    modelCNN  = get_modelparameters(params,img_shape, lr)

    if not data_augmentation:
        print('Not using data augmentation.')
        if early_stopping:
            history       = modelCNN.fit(X_train, y_train, batch_size, epochs, validation_data=[X_val,yval1h], class_weight = class_weight,verbose=1,callbacks=[callbacks],shuffle=True)
        else:
            history = modelCNN.fit(X_train, y_train,batch_size, epochs, validation_data=[X_val,yval1h],class_weight = class_weight,verbose=1,shuffle=True)

    else:
        print('Using real-time data augmentation.')
        aug = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,fill_mode="nearest")

        if early_stopping:
            history       = modelCNN.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=len(X_train) // batch_size, epochs=epochs,validation_data=[X_val,yval1h],
                            class_weight = class_weight,verbose=1,callbacks=[callbacks],shuffle=True)

        else:
            history = modelCNN.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size,epochs=epochs,
                      validation_data=[X_val,yval1h],class_weight = class_weight,verbose=1,shuffle=True)
    return history, modelCNN




def model_save(model, model_name):
    '''
    Function to save the fully trained model

    INPUTS:
        model: Here it will be modelCNN, that is the fully trained network
        model_name: the name of the fully trained network
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open('./FRBID_model/'+model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('./FRBID_model/'+model_name+".h5")
    print("Saved model to disk")            
    return model



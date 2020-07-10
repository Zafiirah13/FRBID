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

import matplotlib.pylab as plt
from keras.models import Model
import numpy as np
import itertools
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import os
from FRBID_code.util import ensure_dir

#plt.rc('text', usetex=True)
#plt.rc('font',**{'family':'DejaVu Sans','serif':['Palatino']})
figSize  = (12, 8)
fontSize = 20


def optimsation_curve(history_,plot_dir1,plot_dir2,show=True):
    '''    
    Function to plot the accuracy and loss during training and validation

    INPUTS:
        history_: The log history of the fully trained model
        plot_dir1: The directory to save accuracy curves
        plot_dir2: The directory to save the loss curves
    '''
    plt.figure(figsize=figSize)
    plt.plot(history_.history['acc'], c='r', lw=2.5, label = 'Training')
    plt.plot(history_.history['val_acc'], c='b', lw=2.5, label = 'validation')
    plt.xlabel('Epoch', fontsize=fontSize)
    plt.ylabel('Accuracy', fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    ensure_dir(plot_dir1)
    plt.savefig(plot_dir1, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figSize)
    plt.plot(history_.history['loss'], c='r', lw=2.5, label = 'Training')
    plt.plot(history_.history['val_loss'], c='b', lw=2.5, label = 'validation')
    plt.xlabel('Epoch', fontsize=fontSize)
    plt.ylabel('Loss', fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    ensure_dir(plot_dir2)
    plt.savefig(plot_dir2, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.close()

def feature_maps(model, x_train, y_train, img_index, ofname,show=True):
    '''
    Function to plot the feature maps of the convolutional layer
    INPUTS:
        x_train: The training set images
        y_train: The label of the training set
        image_index: Choose the index of a single candidate file- values can range from 0 to Nimages
        ofname: The directory to save the feature maps for RFI and FRB
    '''

    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)
        
    # redefine model to output right after the first hidden layer
    redefine_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    feature_maps   = redefine_model.predict(x_train)
    
    # plot 16 maps in an 4x4 squares
    square = 4
    ix     = 1
    plt.figure(figsize=(8,8))

    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[img_index, :, :, ix-1])#, cmap='gray')
            ix += 1
    plt.tight_layout()
    ensure_dir(ofname)
    plt.savefig(ofname, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()



def plot_confusion_matrix(cm, classes_types, ofname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdPu,show=True):#  plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        cm = cm.astype('int')
    

    print(cm)
    plt.figure(figsize=(9,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cb=plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes_types))
    plt.xticks(tick_marks, classes_types, rotation=45)
    plt.yticks(tick_marks, classes_types)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if (cm[i, j] < 0.01) or (cm[i,j] >= 0.75)  else "black",fontsize=18)
        else:
            plt.text(j, i,"{:0}".format(cm[i, j]), horizontalalignment="center",
                    color="white" if (cm[i, j] < 3) or (cm[i,j] >= 100)  else "black",fontsize=18)

    
    plt.ylabel('True label',fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)
    plt.tight_layout()
    ensure_dir(ofname)
    plt.savefig(ofname, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()


def plot_roc(fpr, tpr, auc, ofname, show=True):
    '''
    Function to plot the ROC curves (Receiver Characteristic Curves)
    INPUTS:
        fpr: False Positive rate
        tpr: True Positive Rate
        auc: Area under curve
        ofname: The directory to save the roc curve
    '''

    plt.figure(figsize=figSize)
    plt.plot(fpr, tpr, lw=2.5, label='ROC curve (area = {0:0.2f})'.format(auc))
    plt.xlabel("FPR",fontsize=fontSize)
    plt.ylabel("TPR",fontsize=fontSize)
    plt.ylim([0.0,1.01])
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.15))
    plt.tight_layout()
    ensure_dir(ofname)
    plt.savefig(ofname, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.close()


def plot_images(data, ID, y_true, odir, savefig=False, show=True):
    '''
    Function to plot the input images: DM_time and Frequency_time image

    INPUTS:
        data: The candidate images 
        ID: The transient ID
        y_true: The label of the images
        odir: The directory to save the images
        savefig: True if one want to save the images
    '''
    for j in range(data.shape[0]):
        fig, axs = plt.subplots(1,data.shape[3], figsize=(15, 4), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .2, wspace=.05)
        titles = ['DM-Time','Frequency-Time']    
        axs    = axs.ravel()

        for i in range(1,data.shape[3]+1):
            varray = data[j,:,:,i-1]
            im     = axs[i-1].imshow(varray[:,:])#,cmap='gray')
            cb     = fig.colorbar(im,fraction=0.046, pad=0.04,ax=axs[i-1])
            cb.ax.tick_params(labelsize=14)
            axs[i-1].title.set_text(titles[i-1])
        
        plt.tight_layout() 
        if savefig:
        	ofname = os.path.join(odir, str(y_true[j]), str(ID[j]) + ".pdf")
        	ensure_dir(ofname)
        	plt.savefig(ofname,bbox_inches = 'tight',pad_inches = 0.1)

    if show:
        plt.show()
    


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


import numpy as np
import pandas as pd

import h5py
import matplotlib.pylab as plt
import os
import glob



def _get_data_path():
    path = "./data/"
    return path

def shuffle_all(L, n, seed=0):
    '''INPUT:
    L: List [X,y] for e.g [X_train, y_train]
    n: len of y for e.g len(y_train)

    OUTPUT:
    L: X, y are shuffled for e.g X_train, y_train
    '''
    np.random.seed(seed)
    perm = np.random.permutation(n)
    for i in range(len(L)):
        L[i] = L[i][perm]

    return L

def load_data(csv_files='./data/csv_labels/test_set.csv', data_dir = './data/test_set/',n_images = 'dm_fq_time'):
    ID = []; y = []
    dm_time = [] ; fq_time = []
    
    list_hdf5_filename = os.listdir(data_dir)
    data_csv = pd.read_csv(csv_files)
    
    # Iterate through Number of candidate files in directory
    for i in range(len(list_hdf5_filename)):
        row = data_csv[data_csv.h5.str.match(list_hdf5_filename[i])]
        if (row.shape[0]!=0):
            cand_name = row.h5.values[0]
            label = row.label.values[0]
            ID.append(cand_name)
            y.append(label)
            with h5py.File(data_dir+str(cand_name), 'r') as f:
                dm_t = np.array(f['data_dm_time'])
                fq_t = np.array(f['data_freq_time']).T
                dm_time.append(dm_t); fq_time.append(fq_t)

    dm_time_img = np.expand_dims(np.array(dm_time),1)
    fq_time_img = np.expand_dims(np.array(fq_time),1)

    
    if n_images == 'dm_fq_time':
        X_img = np.stack((dm_time_img,fq_time_img),axis=-1)
        X_img = X_img.reshape(X_img.shape[0], 256, 256, 2)

    if n_images == 'dm_time':
        X_img = dm_time_img.reshape(dm_time_img.shape[0], 256, 256 , 1)


    if n_images == 'fq_time':
        X_img = fq_time_img.reshape(fq_time_img.shape[0], 256, 256 , 1)

    X_img = X_img/255.

    ID = np.array(ID)
    Y = np.array(y).astype(np.int32)
    X_img = X_img.astype(np.float32)
    return X_img, Y, ID

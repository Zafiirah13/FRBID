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
import os
import glob
import h5py
from FRBID_code.util import ensure_dir
from keras.models import model_from_json

#-------------------------------------------------------
#----Load Candidate images and Stacked them for CNN-----
#-------------------------------------------------------



def load_candidate(data_dir = './data/test_set/',n_images = 'dm_fq_time'):
    '''
    Function to select only .hdf5 files from a specified directory and then load them as np.array
    Keeps track of the filename
    INPUTS:
        data_dir (str): The directory that contains hdf5 files
        n_images (str): Can take either 'dm_fq_time' or 'dm_time' or 'fq_time'
    OUTPUT:
        X_img (array): Return candidates pixels in an array with same (N, 256, 256, 2) if n_images='dm_fq_time'
        ID (array): An array of candidate filename that are in that folder
    '''
    ID = []; y = []
    dm_time = [] ; fq_time = []
    
    list_hdf5_filename = os.listdir(data_dir)
    list_hdf5_path = glob.glob(data_dir+'*.hdf5')
     
    for i in range(len(list_hdf5_filename)):
        with h5py.File(str(list_hdf5_path[i]), 'r') as f:
            dm_t = np.array(f['data_dm_time'])
            fq_t = np.array(f['data_freq_time']).T
            dm_time.append(dm_t); fq_time.append(fq_t); ID.append(list_hdf5_filename[i])

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
    X_img = X_img.astype(np.float32)

    return X_img, ID

#---------------------------------------
#-------Prediction of a candidate-------
#---------------------------------------

def FRB_prediction(model_name, X_test, ID, result_dir, probability):
    '''
    The code will load the pre-trained network and it will perform prediction on new candidate file.

    INPUT:
    model_name: 'NET1_32_64', 'NET1_64_128', 'NET1_128_256', 'NET2', 'NET3'
    X_test : Image data should have shape (Nimages,100,100,3), (Nimages,30,30,3), (Nimages,30,30,4). This will vary depending on the criteria one use for min_pix, max_pix and num_images.
    ID: The transient ID extracted from the csv file ID=data.iloc[:,0]
    result_dir: The directory to save the csv prediction file

    OUTPUT:
    overall_real_prob: An array of probability that each source is real. Value will range between [0 to 1.0]
    overall_dataframe: A table with column transientid of all sources and its associated probability that it is a real source
    '''
    # load json and create model
    json_file = open("./FRBID_model/"+model_name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fit_model  = model_from_json(loaded_model_json)

    # load weights into new model
    fit_model.load_weights("./FRBID_model/"+model_name+".h5")
    print("Loaded model:"+ model_name +" from disk")

    # Overall prediction for the whole sample
    overall_probability = fit_model.predict_proba(X_test)

    # For all the candidate, output the probability that it is a real source
    overall_real_prob = overall_probability[:,1]
    overall_dataframe = pd.DataFrame(ID, columns=['candidate'])
    overall_dataframe['probability'] = overall_real_prob
    overall_dataframe['label'] = np.round(overall_real_prob>=probability)
    ensure_dir(result_dir)
    overall_dataframe.to_csv(result_dir+'results_'+model_name+'.csv',index=None)
    return overall_real_prob, overall_dataframe

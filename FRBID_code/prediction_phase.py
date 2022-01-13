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
import h5py
from FRBID_code.util import ensure_dir
from FRBID_code.load_data import generate_dm_time, get_parameters
from FRBID_code.load_data import get_masked_bands, reconstruct_normal
from keras.models import model_from_json
from os import path
from os import listdir
from glob import glob

#-------------------------------------------------------
#----Load Candidate images and Stacked them for CNN-----
#-------------------------------------------------------


def load_candidate(data_dir = './data/test_set/',n_images = 'dm_fq_time', cands_csv="candidates.csv"):
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
    ID = []
    dm_time = [] ; fq_time = []
    cand_info = pd.read_csv(cands_csv)

    hdf5_files = hdf5_files = glob(path.join(data_dir, "*.hdf5"))
     
    freq_time_tmp = np.empty((256, 256), dtype=np.float32)

    for idx, hdf5_file in cand_info.iterrows():

        file_path = path.join(data_dir, hdf5_file["hdf5"])

    #for hdf5_file in hdf5_files:
    #    row = cand_info[cand_info.hdf5.str.match(path.basename(hdf5_file))]
    #    if (row.shape[0] != 0):
        with h5py.File(file_path, 'r+') as hf:
            params = get_parameters(hdf5_file, hf)

            if "/cand/ml/old/dm_time" in hf:
                dm_t = np.array(hf["/cand/ml/dm_time"])
            # We have not processed this file yet
            # Generate the DM-time plane and save it into the archive,
            # so that we can use it afterwards
            else:
                freq_time = np.array(hf["/cand/ml/freq_time"])
                ft_mask = get_masked_bands(freq_time)
                freq_time = reconstruct_normal(freq_time, ft_mask)
                
                delta_dm, dm_t = generate_dm_time(freq_time, params, freq_time_tmp)
                old_ml_group = hf.create_group("/cand/ml/old")
                old_ml_group.attrs["label"] = hf["/cand/ml"].attrs["label"]
                old_ml_group.attrs["prob"] = hf["/cand/ml"].attrs["prob"]
                old_dm_time_dataset = old_ml_group.create_dataset("dm_time", data=hf["/cand/ml/dm_time"])
                old_freq_time_dataset = old_ml_group.create_dataset("freq_time", data=hf["/cand/ml/freq_time"])
                hf["/cand/ml/freq_time"][...] = freq_time
                hf["/cand/ml/dm_time"][...] = dm_t
                cand_dm = hf["/cand/detection"].attrs["dm"]
                hf["/cand/ml/"].attrs.create("dm_range", np.array([(cand_dm - delta_dm, cand_dm + delta_dm, "pc cm^-3")], dtype=np.dtype([("start", np.float), ("end", np.float), ("unit", "S8")])))

            fq_t = np.array(hf['/cand/ml/freq_time'])

            dm_time.append(dm_t)
            fq_time.append(fq_t)
            ID.append(hdf5_file["hdf5"])

    dm_time_img = np.expand_dims(np.array(dm_time),1)
    fq_time_img = np.expand_dims(np.array(fq_time),1)

    if n_images == 'dm_fq_time':
        X_img = np.stack((dm_time_img,fq_time_img),axis=-1)
        X_img = X_img.reshape(X_img.shape[0], 256, 256, 2)

    if n_images == 'dm_time':
        X_img = dm_time_img.reshape(dm_time_img.shape[0], 256, 256 , 1)


    if n_images == 'fq_time':
        X_img = fq_time_img.reshape(fq_time_img.shape[0], 256, 256 , 1)

    ID = np.array(ID)
    X_img = X_img.astype(np.float32)

    return X_img, ID

#---------------------------------------
#-------Prediction of a candidate-------
#---------------------------------------

def FRB_prediction(model_name, X_test, ID, result_dir, probability, data_dir):
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

    for _, cand in overall_dataframe.iterrows():

        with h5py.File(path.join(data_dir, cand["candidate"]), "r+") as hf:

            hf["/cand/ml"].attrs["prob"] = np.array([cand["probability"]], dtype="<f2")
            hf["/cand/ml"].attrs["label"] = np.array([cand["label"]], dtype="<f2")

    overall_dataframe.to_csv(path.join(result_dir, 'results_' + model_name + '.csv'),index=None)
    return overall_real_prob, overall_dataframe

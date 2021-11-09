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
from os import path

def _get_data_path():
    path = "./data/"
    return path

def pad_freq_tm(L):
    '''INPUT:
    L: input freq_time plot 256x256

    Pad the data with noise such that while doing
    the rolling operation the peak does not overlap.

    Based on time averaging calculations for the imgae and L-band,
    1 pulsewidth = 2 timesamples 

    If the fudge_factor = 50 

    OUTPUT:
    L: output freq_time plot 256x512
    '''

    mu, sigma = np.mean(L, axis=1), np.std(L, axis=1)
    freq_time1 = np.random.normal(mu, sigma, [512,256]).T
    freq_time1[:, 128:384] = L

    return freq_time1

def get_parameters(row, h5_data):
    '''INPUT:
    
    From the CSV file get the information regarding
    ftop_mhz : highest frequency in the MHz
    chan_band_mhz : Channel resolution in MHz
    nchan : Number of channels

    OUTPUT:
    f_bot : bottom freq
    bw : bandwidth
    '''

    tsamp_s = row.tsamp_s.values[0]
    ftop_mhz = row.ftop_mhz.values[0]
    chan_band_mhz = row.chan_band_mhz.values[0]
    nchan = row.nchan.values[0]
    o_nchan = 256          #output data has 256 channel
    freq_avg = nchan/o_nchan

    freq_top_mhz = ftop_mhz - (chan_band_mhz / 2) * (1 - freq_avg)
    avg_chan_band_mhz = chan_band_mhz*(freq_avg)
    freq_bot_mhz =  freq_top_mhz +  avg_chan_band_mhz*(o_nchan - 1) 
    
    # sampling time is in SECONDS
    # pulse width is in MILLISECONDS
    width_ms = h5_data["/cand/detection"].attrs["width"]
    # Get the pulse width in the raw samples
    # NOTE: pulse width is converted to seconds in this equation
    width_samp = int(round((width_ms * 1e-03) / tsamp_s))
    # We average the data so that we have 2 samples across the pulse width
    # As discussed, these 2 samples are theoretical as the single-pulse detection
    # often underestimates the width
    time_avg = int(width_samp / 2) if width_samp > 1 else 1
    # And that's the averaging time of the ML data (the DM-time and freq-time planes)
    # This is in SECONDS
    ml_sampling_time = tsamp_s * time_avg

    params = {
            "ftop_mhz"      : freq_top_mhz,
            "fbot_mhz"      : freq_bot_mhz,
            "chan_band_mhz" : avg_chan_band_mhz,
            "tsamp_s"       : ml_sampling_time,
            "width_ms"      : width_ms,
            "cand_dm"       : h5_data["/cand/detection"].attrs["dm"]
    }

    return params

def get_dm_range(params):
    '''
    INPUT:
    L: input freq_time plot

    OUTPUT:
    L: output dm_time plot
    '''
    # Determine the DM trial range(delta_dm)
    k_dm = 4148.808
    freq_hi  = params['ftop_mhz']
    freq_lo  = params['fbot_mhz']
    width_ms = params['width_ms']
    cand_dm = params['cand_dm']

    # Get exccess DM worth of 30*pulse_width
    fudge_factor = 30      #approximatly 18%
    delta_dm = (fudge_factor*(width_ms/1000)) / (k_dm*(freq_lo**-2 - freq_hi**-2))

    if delta_dm > cand_dm:
      delta_dm = cand_dm

    return delta_dm

def generate_dm_time(L, params, freq_time):
    '''INPUT:
    L: input freq_time plot
    

    OUTPUT:
    L: output dm_time plot
    '''
    
    k_dm = 4148.808
    freq_hi  = params['ftop_mhz']
    freq_lo = params["fbot_mhz"]
    tsamp_s = params['tsamp_s']
    
    delta_dm = get_dm_range(params) #get DM offset propotional to pulse width
    # Make it (256, 256), extract the right chunk from freq_time
    dm_time  = np.empty((256,256))

    freq_time_backup = pad_freq_tm(L) #pad the data so to not overflow in the image
    const_scaling = k_dm / tsamp_s
    freq_shifts = (1 / np.linspace(freq_hi, freq_lo, 256)**2 - 1 / freq_hi**2) * const_scaling
    
    j_dm=0
    for dm_offset in np.linspace(-delta_dm, delta_dm, 256):
        chan_shifts = np.round(freq_shifts * dm_offset).astype(int)
        for i_chan in np.arange(256):
            shift_chan = chan_shifts[i_chan]
            freq_time[i_chan, :] = freq_time_backup[i_chan, 128 + shift_chan : 384 + shift_chan ]
            # freq_time[i_chan, :] = np.roll(freq_time[i_chan, :], int((-1)*round(shift_chan)), axis=0) #because we are shifting the dedispersed signal
        dm_time[(j_dm), :] = freq_time.sum(axis=0)
        j_dm = j_dm + 1
    return dm_time

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

def load_data(csv_files='./train_set.csv', data_dir = './data/train/', n_images = 'dm_fq_time'):
    ID = []; y = []
    dm_time = [] ; fq_time = []
    
    list_hdf5_filename = os.listdir(data_dir)
    data_csv = pd.read_csv(csv_files)
    
    print("Loading files...")

    freq_time_tmp = np.empty((256, 256))

    for hdf5_file in list_hdf5_filename:
        row = data_csv[data_csv.hdf5.str.match(hdf5_file)]
        if (row.shape[0] != 0):
            cand_name = hdf5_file
            label = row.label.values[0]
            ID.append(cand_name)
            y.append(label)
            print(hdf5_file)
            with h5py.File(path.join(data_dir, cand_name), "r+") as hf:
                params = get_parameters(row, hf)

                # We have already reprocessed this archive
                # Just read the existing DM-time plane to save time
                if "/cand/ml/old/dm_time" in hf:
                    dm_t = np.array(hf["/cand/ml/dm_time"])
                # We have not processed this file yet
                # Generate the DM-time plane and save it into the archive,
                # so that we can use it afterwards
                else:
                    dm_t = generate_dm_time(hf['/cand/ml/freq_time'], params, freq_time_tmp)
                    old_ml_group = hf.create_group("/cand/ml/old")
                    old_dm_time_dataset = old_ml_group.create_dataset("dm_time", data=hf["/cand/ml/dm_time"])
                    old_freq_time_dataset = old_ml_group.create_dataset("freq_time", data=hf["/cand/ml/freq_time"])
                    hf["/cand/ml/dm_time"][...] = dm_t

                fq_t = np.array(hf['/cand/ml/freq_time'])

                print(dm_t.shape)
                print(fq_t.shape)

                dm_time.append(dm_t)
                fq_time.append(fq_t)

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
    Y = np.array(y).astype(np.int32)
    X_img = X_img.astype(np.float32)

    return X_img, Y, ID

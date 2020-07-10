
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

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import classification_report_imbalanced
from keras.utils import np_utils
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix,balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score
from keras.models import model_from_json
from FRBID_code.plot import plot_confusion_matrix, plot_roc, optimsation_curve, feature_maps, plot_images
import pandas as pd
from FRBID_code.util import ensure_dir



def model_prediction(fit_model, odir, model_name, X_test, y_test, classes=["RFI" , "FRB"], cm_norm=False,load_model=False,show=True):
    '''
    Function to evaluate the trained model

    INPUTS:
        fit_model: if load_model is False, it will fit the existing model that just trained, for e.g modelCNN, Else, it should be NONE
        odir: The directory to save the plots
        model_name: if load_model is True, model_name = 'NET1_32_64', 'NET1_64_128', 'NET1_128_256', 'NET2', 'NET3', Either of them
        X_test, y_test: Evaluate the trained model on a sample of test set having images and its label
        classes: List with the names of the classes considered. Used to label confusion matrix. 
        cm_norm: True if we want the conf_matrix to be betwwen 0 to 1 or False if we want the number of samples correctly classified
        load_model: True if we want to use an already pre-trained model, else False

    OUTPUTS:
        ypred: An array of prediction for the test set array[[0 1 0 0 1 ....]]
        balanced_accuracy, MCC, conf_mat: The metrics  values when evaluating the trained model 
        misclassified: An array of indices from the test set indices that indicates which indices (images) got misclassified
        fit_model: return the train model
        correct_classification: An array of indices from the test set indices that indicates which indices (images) are correctly classified
        probability: The overall probability of each candidate varies betwwen 0 to 1. For a candidate, it outputs prob = [0.1, 0.9], this
                     candidate is therefore an FRB/single Pulse candidate with prob 0.9 and has a probability of 0.1 that it is an RFI.
    '''
    if load_model:
        # load json and create model
        json_file = open("./FRBID_model/"+model_name+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        fit_model  = model_from_json(loaded_model_json)
        # load weights into new model
        fit_model.load_weights("./FRBID_model/"+model_name+".h5")
        print("Loaded model from disk")

    else:
        fit_model = fit_model
    
    ypred          = np.argmax(fit_model.predict(X_test),axis=1)
    probability    = fit_model.predict_proba(X_test)
    accuracy       = accuracy_score(y_test, ypred)
    MCC            = matthews_corrcoef(y_test, ypred)
    conf_mat       = confusion_matrix(y_test, ypred)
    balance_accuracy = balanced_accuracy_score(y_test, ypred)
    fpr, tpr, thres  = roc_curve(y_test, probability[:,1], pos_label=1)

    le             = LabelEncoder()
    labels         = le.fit_transform(y_test)
    yTest          = np_utils.to_categorical(labels,len(classes))
    auc            = roc_auc_score(yTest,probability)    
    misclassified  = np.where(y_test != ypred)[0]
    correct_classification = np.where(y_test == ypred)[0]

    plot_confusion_matrix(conf_mat, classes_types=classes, ofname=os.path.join(odir, "confusion_matrix.pdf"), normalize=cm_norm,show=show)    
    plot_roc(fpr, tpr, auc, ofname=os.path.join(odir, "ROC.pdf"),show=show)
     
    name_file = open(os.path.join(odir, "Results.txt"), 'w')
    name_file.write('='*80+'\n')
    name_file.write('******* Testing Phase for ' + str(classes) + ' *******\n')
    name_file.write('='*80+'\n')
    name_file.write("Accuracy: "                    + "%f" % float(accuracy) + '\n')
    name_file.write("Mathews Correlation Coef: "    + "%f" % float(MCC)      + '\n')
    name_file.write("Balanced Accuracy: "    + "%f" % float(balance_accuracy)      + '\n')
    name_file.write('='*80+'\n')
    name_file.write('='*80+'\n')
    name_file.write('Classification Report\n')
    name_file.write('='*80+'\n')
    name_file.write(classification_report(y_test, ypred, target_names = classes)+'\n')
    name_file.write('='*80+'\n')
    name_file.write('='*80+'\n')
    name_file.write('Classification Report using imbalanced metrics\n')
    name_file.write('='*80+'\n')
    name_file.write(classification_report_imbalanced(y_test, ypred, target_names = classes)+'\n')
    name_file.write('='*80+'\n')
    name_file.close()
        
    return ypred, balance_accuracy, MCC, conf_mat, misclassified, fit_model, correct_classification, probability

def save_classified_examples(X_test, y_test, ID_test, correct_classification, probability,odir_real,
                            odir_bogus,savecsv = True):
    '''
    Function to save the overall probability of each source in csv files

    INPUTS:
        X_test, y_test: Test candidates having images and its associated labels
        ID_test: The transient id for each candidate
        correct_classification: An array of indices from the test set indices that indicates which indices (images) are correctly classified
        probability: The overall probability of each candidate varies betwwen 0 to 1. For a candidate, it outputs prob = [0.1, 0.9], this
                     candidate is therefore a real candidate with prob 0.9 and has a probability of 0.1 that it is bogus
        odir_real: The directory to save the csv file for real candidate
        odir_bogus: The directory to save the csv file for bogus candidate
        savecsv: True to save the csv
    '''
    overall_probability_real = pd.DataFrame(ID_test,columns=['transientid'])
    overall_probability_real['ML_PROB_FRB'] = probability[:,1]


    correct_classification_array = correct_classification
    y_true_correctly_classified = y_test[correct_classification_array]
    ID_correctly_classified = ID_test[correct_classification_array]
    correctly_classified_img   = X_test[correct_classification_array]
    prob_correctly_classified = probability[correct_classification_array]

    # select the real and bogus indices that were correctly classified
    bogus_true_indices = correct_classification_array[y_true_correctly_classified==0]
    real_true_indices = correct_classification_array[y_true_correctly_classified==1]

    # Assign probability to a source being real or bogus
    prob_bogus = probability[bogus_true_indices,0]
    prob_real = probability[real_true_indices,1]

    # Select the transient ID of real and bogus that were correctly classified
    ID_bogus = ID_test[bogus_true_indices]
    ID_real = ID_test[real_true_indices]

    # Select the image array of real and bogus that were correctly classified
    correctly_cfd_real_img = X_test[real_true_indices]
    correctly_cfd_bogus_img = X_test[bogus_true_indices]

    # Create a DataFrame to store the transient ID and Probability of each source in separate csv file and 
    # save then in different directory
    correctly_classified_bogus = pd.DataFrame(ID_bogus,columns=['transientid'])
    correctly_classified_bogus['ML_PROB'] = prob_bogus

    correctly_classified_real = pd.DataFrame(ID_real,columns=['transientid'])
    correctly_classified_real['ML_PROB'] = prob_real

    if savecsv:

        ofname_real = os.path.join(odir_real)
        ensure_dir(ofname_real)
        overall_probability_real.to_csv(ofname_real+'probability_candidate_classified_as_frb.csv', index=None)

        ofname_real = os.path.join(odir_real)
        ensure_dir(ofname_real)
        correctly_classified_real.to_csv(ofname_real+'correctly_classified_frb.csv', index=None)
    
        ofname_bogus = os.path.join(odir_bogus)
        ensure_dir(ofname_bogus)
        correctly_classified_bogus.to_csv(ofname_bogus+'correctly_classified_rfi.csv', index=None)
    return overall_probability_real, correctly_classified_bogus, correctly_classified_real


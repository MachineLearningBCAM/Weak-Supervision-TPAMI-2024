#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 07:37:44 2024
"""

import logging
from pathlib import Path

import numpy as np
from scipy.io import savemat
import scipy.io as sio

from wrench.dataset import load_image_dataset
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import MMP
from wrench.labelmodel import GenerativeModel# MajorityVoting
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# from .majority_voting import MajorityVoting

# from wrench.labelmodel import BalsubramaniFreund

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset

data = sio.loadmat('./datasets/basketball.mat')
train_data = [data['train_pred'], data['train_labels']]
valid_data = [data['val_pred'], data['validation_labels']]

reps = 10
n, _ = train_data[0].shape
predict_probabilities = np.zeros((reps, n))
predictions = np.zeros((reps, n))
time_total = np.zeros((reps))

ci_class0_prob = np.zeros((reps), dtype = object)
ci_class1_prob = np.zeros((reps), dtype = object)
m_prob = np.zeros((reps), dtype = object)
observed_accuracy_prob = np.zeros((reps), dtype = object)


for i in range(reps):
    #### Run label model: BalsubramaniFreund
    try:
        label_model = MMP(include_dis = True)
        label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    except Exception:
        continue
              
    ci_class0_prob[i], ci_class1_prob[i], m_prob[i], observed_accuracy_prob[i] = label_model.get_confidence(train_data, "pattern", n_weak_disagree = 8)#n_weak_disagree is the number of weak classifiers where we allow disagreement to obtain the patterns
    
    if i == 0:
        ci_class0_prob_mean = ci_class0_prob[i]
        ci_class1_prob_mean = ci_class1_prob[i]
        m_prob_mean = m_prob[i]
        observed_accuracy_prob_mean = observed_accuracy_prob[i]
     
    else:
      
        
        ci_class0_prob_mean += ci_class0_prob[i]
        ci_class1_prob_mean += ci_class1_prob[i]
        m_prob_mean += m_prob[i]
        observed_accuracy_prob_mean += observed_accuracy_prob[i]
        
    Y_p = label_model.predict_proba(train_data)
    
    predict_probabilities[i, :] = Y_p[:, 1]
    
    predictions[i, :] = np.around(Y_p[:, 1])
    
    
    del label_model
    
ci_class0_prob_mean = ci_class0_prob_mean/reps
ci_class1_prob_mean = ci_class1_prob_mean/reps
m_prob_mean = m_prob_mean/reps
observed_accuracy_prob_mean = observed_accuracy_prob_mean/reps

prob_mean = np.mean(predict_probabilities, axis = 0)
prob_std = np.std(predict_probabilities, axis = 0)

st = np.mean(prob_std)
Y_p_class0 = 1-prob_mean
pred_mean = np.round(prob_mean)

true_labels = np.squeeze(train_data[1])

brier_score = brier_score_loss(true_labels,prob_mean)

logloss = log_loss(true_labels,prob_mean)

mistakes = 0
for i in range(len(pred_mean)):
    if int(pred_mean[i]) != true_labels[i]:
        mistakes += 1

error = mistakes/len(pred_mean)

f1 = f1_score(true_labels, pred_mean)


mistakes_list = np.zeros((1, len(true_labels)));
for j in range(len(true_labels)):
    if true_labels[j] != pred_mean[j]:
        mistakes_list[0, j] = 1
                         
y_calibration, x_calibration = calibration_curve(np.squeeze(train_data[1]), np.clip(prob_mean, 0, 1), n_bins=10)

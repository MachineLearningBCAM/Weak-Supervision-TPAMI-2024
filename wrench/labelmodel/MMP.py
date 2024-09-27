#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 07:34:18 2024
"""

import logging
import warnings
from typing import Any, Optional, Union

import cvxpy as cp
import numpy as np
import scipy as sp
from statsmodels.stats.proportion import proportion_confint

from ..evaluation import METRIC
from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels
from .majority_voting import MajorityVoting
from sklearn.model_selection import train_test_split
import sys

logger = logging.getLogger(__name__)

ABSTAIN = -1

class MMP(BaseLabelModel):
    def __init__(self,
                 solver: Optional[str] = 'SCS',
                 cp_verbose: Optional[bool] = False,
                 **kwargs: Any):
        super().__init__()
        # self.hyperparas = {}
        self.solver = solver
        self.cp_verbose = cp_verbose

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            majority_vote: Optional[np.ndarray] = False,
            unsupervised: Optional[np.ndarray] = False,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            acc_bound: Optional[float] = 0.3,
            signif_lvl: Optional[float] = 0.05,
            n_labeled: Optional[int] = 100,
            ci_name: Optional[str] = 'wilson',
            verbose: Optional[bool] = False,
            weak: Optional[int] = None,
            n_weaks: Optional[int] = None,
            seed: Optional[int] = None,
            random_guess: Optional[int] = None,
            include_dis: Optional[np.ndarray] = False,
            **kwargs: Any):
        
        self.majority_vote = majority_vote

        # self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class

         L = dataset_train[0]
        
        if majority_vote == True:
            label_model = MajorityVoting()
            label_model.fit(dataset_train)
            Y_p = label_model.predict_proba(dataset_train)

            pred_majority_vote = np.around(Y_p[:, 1])
            L2 = np.expand_dims(pred_majority_vote, 1)
            L = np.hstack((L, L2))
                
        n_class = n_class or round(L.max()) + 1
        self.n_class = n_class

        L_aug = self._initialize_L_aug(L)

        # count number of predictions for every classifier
        n_preds = np.sum(L_aug, axis=(-2, -1))
        self.n_preds = n_preds


        # classifier count, number of datapoints, number of classes
        p, n, _ = L_aug.shape
    

        param_probs = np.zeros((3, p))
        # estimate class frequencies and classifier parameters (e.g. accuracies)
        if unsupervised == False:
            if y_valid is None:
                y_valid = np.squeeze(dataset_valid[1])
                # y_valid = np.array(dataset_valid.labels)
            L_val = dataset_valid[0]
            L_val, X_test, y_valid, Y_test = train_test_split(L_val, y_valid, test_size=None, train_size=n_labeled)
            dataset_valid = [L_val, y_valid]
            if majority_vote == True:
                label_model = MajorityVoting()
              
                label_model.fit(dataset_valid)
                Y_p = label_model.predict_proba(dataset_valid)
                pred_majority_vote = np.around(Y_p[:, 1])
                L2 = np.expand_dims(pred_majority_vote, 1)
                L_val = np.hstack((L_val, L2))
    
            L_val_aug = self._initialize_L_aug(L_val)
    
    
            n_val_preds = np.sum(L_val_aug, axis=(-2, -1))
            
            st = np.zeros((n_val_preds[0], p))
            for j in range(p):
            #
                param_probs[1, j] = METRIC['acc'](y_valid, L_val_aug[j])
            

            for j in range(p):
                for nn in range(n_val_preds[0]):
                    if int(L_val[nn, j]) == y_valid[nn]:
                         st[nn, j] = 1
                    else:
                         st[nn, j] = 0
    
            a =np.std(st, axis =0)/np.sqrt(n_val_preds[0])
            label_model = MajorityVoting()
          
            label_model.fit(dataset_train)
            Y_MV = label_model.predict(dataset_train)
            
            b = np.zeros((p, ))
            
            for j in range(p):
            #
                b[j] = METRIC['acc'](Y_MV, L_aug[j])
            
            for j in range(p):
                a[j] = max(a[j], np.abs(param_probs[1, j] - b[j]))# np.abs(param_probs[1, :] - b)# 1/np.sqrt(n_val_preds[0])# 
            
            param_probs[2, :] = param_probs[1, :] + a
            param_probs[0, :] = param_probs[1, :] - a
        
            
            if include_dis == True:
                
                d, d_upper, d_lower = self._discrepancy(L, dataset_train, acc_bound)
                
                param_probs2 = np.zeros((3, len(d)))
                
                param_probs2[1, :] = d
                
             
                param_probs2[2, :] = d_upper
                
                param_probs2[0, :] = d_lower
                
                param_probs = np.concatenate((param_probs, param_probs2), axis = 0)
                
                
            
        elif unsupervised == True:
             
            d, d_upper, d_lower = self._discrepancy(L, dataset_train, acc_bound)
         
         
            param_probs[1, :] = d
            param_probs[2, :] = d_upper
            param_probs[0, :] = d_lower
         
             
        else:
            print("Error")
            sys.exit()


        param_probs = np.clip(param_probs, 0, 1)

        param_cts = np.multiply(n_preds, param_probs)
        
        param_cts = np.nan_to_num(param_cts)
        
        param_probs[0, :] = np.nan_to_num(param_probs[0, :])
        param_probs[1, :] = np.nan_to_num(param_probs[1, :])
        param_probs[2, np.isnan(param_probs[2, :])] = 1
        self.param_probs = param_probs
        
        

        param_eps = np.maximum(param_cts[1, :] - param_cts[0, :],
                param_cts[2, :] - param_cts[1, :])
        
        if unsupervised == False:
            
            for j in range(p):
                label_model = MajorityVoting()
               
                label_model.fit(dataset_valid)
                Y_p = label_model.predict_proba(dataset_valid)
                pred_majority_vote = np.around(Y_p[:, 1])
                acc_true_mv =  METRIC['acc'](pred_majority_vote, np.round(L_val_aug[j]))
                param_eps[j] = max(param_eps[j], abs(param_probs[1, j] - acc_true_mv)*(n_preds[j]))

        
        
        self.param_eps = np.nan_to_num(param_eps)
        self.param_cts = param_cts
        # make convex program
        self.prob, sigma = self._make_cp(L_aug, p, param_cts,
                param_eps)

        # solve convex program
        self.prob.solve(solver=self.solver, verbose=self.cp_verbose)

        self.param_wts = sigma.value

    def _make_cp(self, L_aug, p, param_cts, param_eps):
        # create variables
        sigma = cp.Variable(p)

        # create objective
        aggregated_weights = self._aggregate_weights(L_aug, sigma)
        obj = cp.Maximize(sigma @ param_cts[1, :]
                - cp.abs(sigma) @ param_eps
                - cp.sum(cp.log_sum_exp(aggregated_weights, axis = 1)))

        # create constraints
        constrs = []

        return cp.Problem(obj, constrs), sigma

    def _initialize_L_aug(self, L):
        L = L.T
    
        L_aug = (np.arange(self.n_class) == L[..., None]).astype(int)

        return L_aug

    def _aggregate_weights(self, L_aug, param_wts, mod=cp):
        # assuming param_wts is a k by k matrix where element ij is the weight
        # associated with the classifier predicting j when true label is i.
        p, n, _ = L_aug.shape

        Y_p = mod.multiply(L_aug[0], param_wts[0])

        for j in range(1, p):
            # pick out column of confusion matrix (since we see observed pred)
            # for every datapoint.  Resulting matrix is n by k
            Y_p += mod.multiply(L_aug[j], param_wts[j])
            # for confusion matrices where param_wts is shape (p, k, k)
           
        return Y_p

    def _make_bf_preds(self, L_aug, param_wts):
        Y_p = self._aggregate_weights(L_aug, param_wts, mod=np)
        return sp.special.softmax(Y_p, axis=1)
    
    def _discrepancy(self, L, dataset, acc_bound):
        
        label_model = MajorityVoting()
        label_model.fit(dataset)
        Y_p = label_model.predict_proba(dataset)
        
        y = np.argmax(Y_p, axis = 1)
        y = y.astype(int)
        

        n_weaks = len(L[0, :])
        d = np.zeros((n_weaks))
        epsilon = np.zeros((n_weaks))
        for i in range(n_weaks):
            s2 = 0
            for j in range(len(y)):
                s = 0
                for k in range(self.n_class):
                    if L[j, i] == -1:
                        h = 1/2
                    else:
                        if L[j, i] == k:
                            h = 1
                        else:
                            h = 0
                    if y[j] == k:
                        h_mv = 1
                    else:
                        h_mv = 0
                    s += h*h_mv
                s2 += s
            d[i] = (1/len(y))*(s2)# +acc_bound


            epsilon[i] = np.std(L[:, i])/np.sqrt(len(L[:, 0]))# max(np.std(L[:, i])/np.sqrt(len(L[:, 0])), abs(d[i] - acc_true_mv))
    
        d_upper = np.clip(d + epsilon, 0, 1)# p[1]#np.clip(d + epsilon, 0, 1)
        d_lower = np.clip(d - epsilon, 0, 1)# p[0]#np.clip(d - epsilon, 0, 1)
        
        return d, d_upper, d_lower


    def predict_proba(self,
            dataset: Union[BaseDataset, np.ndarray], weak: Optional[int] = None, n_weaks: Optional[int] = None, random_guess: Optional[int] = None, seed: Optional[int] = None,
            **kwargs: Any) -> np.ndarray:
        # L = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        L = dataset[0]
        if self.majority_vote == True:
            label_model = MajorityVoting()
           
            label_model.fit(dataset)
            Y_p = label_model.predict_proba(dataset)
            pred_majority_vote = np.around(Y_p[:, 1])
            L2 = np.expand_dims(pred_majority_vote, 1)
            L = np.hstack((L, L2))
        L_aug = self._initialize_L_aug(L)
        Y_p = self._make_bf_preds(L_aug, self.param_wts)
        return Y_p
    

    
    def _get_prediction_patterns(self, L, n_weak_disagree):
        # want each prediction to be in a row
        L = L.T
        n_instances = L.shape[1]
        idx2 = []
        for i in range(30):# n_instances):## We are not computing patterns for all instances
            l = []
            l.append(i)
            for j in range(i+1, n_instances):
                idx = np.where(L[:, i]!=L[:, j])
                x = len(idx[0])
                if x <= n_weak_disagree:
                    l.append(j)

                        
            if len(l)>0:
                if len(idx2) == 0:
                    idx2.append(np.unique(l))
                else:
                    a = 0
                    for k in range(len(idx2)):
                        if set(np.unique(l)).issubset(idx2[k]):
                            a += 1
                    if a == 0:
                        idx2.append(np.array(np.unique(l)))
        
        self.n_uniq_cols2 = len(idx2)
        self.idx_patterns = np.array(idx2)
        
    def _get_pattern_LF(self, L, n_weak_disagree):
        # want each prediction to be in a row
        L = L.T
        n_instances = L.shape[1]
        idx2 = np.zeros((self.n_class, 1), dtype = object)
       
        for j in range(self.n_class):
            l = []
            for i in range(n_instances):# n_instances):## We are not computing patterns for all instances
                idx = np.where(L[:, i] == j)
                if len(idx[0]) > n_weak_disagree:
                    l.append(i) 
            idx2[j, 0] = l
                
        self.n_uniq_cols2 = 1
        self.idx_patterns = idx2
        
    def _get_prob_patterns(self, Y_p):
        
        prob_values = [0.90, 0.8, 0.5, 0, 0,0, 0, 0, 0]
        r = 20# len(prob_values)
        
        idx2 = np.zeros((self.n_class, r), dtype = object)
        for j in range(r):
            for i in range(self.n_class):
                # perc = np.percentile(Y_p[:, i], prob_values[j])
                # idx = np.where(Y_p[:, i] >= perc)
                # idx2[i, j] = idx
                if j == 0:
                    my_boolean_arr = (Y_p[:, i] >= 0.95)
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 1:
                    
                    my_boolean_arr = (Y_p[:, i] >= 0.9) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                
                elif j == 2:
                    
                    my_boolean_arr = (Y_p[:, i] >= 0.85) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                    
                elif j == 3:
                    
                    my_boolean_arr = (Y_p[:, i] >= 0.80) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                    
                elif j == 4:
                        
                    my_boolean_arr = (Y_p[:, i] >= 0.75) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 5:
                            
                    my_boolean_arr = (Y_p[:, i] >= 0.70) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                    
                elif j == 6:
                                
                    my_boolean_arr = (Y_p[:, i] >= 0.65) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                    
                elif j == 7:
                                    
                    my_boolean_arr = (Y_p[:, i] >= 0.60) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 8:
                                    
                    my_boolean_arr = (Y_p[:, i] >= 0.55) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 9:
                                    
                    my_boolean_arr = (Y_p[:, i] >= 0.50) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                    
                elif j == 10:
                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.50) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 11:
                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.45) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 12:
                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.40) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 13:
                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.35) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 14:
                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.30) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 15:
                                        
                    my_boolean_arr = (Y_p[:, i] <= 0.25) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 16:
                                        
                    my_boolean_arr = (Y_p[:, i] <= 0.2) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 17:
                                            
                    my_boolean_arr = (Y_p[:, i] <= 0.15) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 18:
                                                
                    my_boolean_arr = (Y_p[:, i] <= 0.1) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
                elif j == 19:
                                                    
                    my_boolean_arr = (Y_p[:, i] <= 0.05) # (Y_p[:, i] <= 0.95) & 
                    idx = np.where(my_boolean_arr == True)
                    idx2[i, j] = idx
        self.n_uniq_cols2 = r
        self.idx_patterns = idx2
        
    def _get_prob_calibration(self, Y_p):
        
        prob_values = [0.95, 0.85, 0.75]
        r = len(prob_values)
        
        idx2 = np.zeros((self.n_class, r), dtype = object)
        for j in range(r):
            for i in range(self.n_class):
                if j == 0:
                    idx = np.where(Y_p[:, i] >= prob_values[j])
                    idx2[i, j] = idx
                else:
                    idx = np.where(Y_p[:, i] <= prob_values[j-1])#  & Y_p[:, i] >= prob_values[j])
                    idxxx = np.where(Y_p[:, i] >=prob_values[j])
                    idx = np.intersect1d(idx, idxxx)
                    # idx = np.where(Y_p[:, i] >= prob_values[j])
                    idx2[i, j] = np.array(idx)
                
        

        self.n_uniq_cols2 = r
        self.idx_patterns = idx2
        
    def get_confidence(self, data, patterns, n_weak_disagree):
        # L = check_weak_labels(data)#, n_weaks=n_weaks, random_guess=random_guess)
        # L = check_weak_labels(data)
        L = data[0]
        tl = data[1]

        Y_p = self.predict_proba(data)
        if patterns == "probabilities":
            self._get_prob_patterns(Y_p)
        if patterns == "pattern":
            self._get_pattern_LF(L, n_weak_disagree)
            # self._get_prob_patterns(Y_p)
        elif patterns == "calibration":
            self._get_prob_calibration(Y_p)
        elif patterns == "weak":
            self._get_prediction_patterns(L, n_weak_disagree)
        else:
            print('Error')
            sys.exit()
        
        L_aug = self._initialize_L_aug(L)
        p, n, _ = L_aug.shape
        
        ci = np.zeros((4, self.n_uniq_cols2))
        ci_class0 = np.zeros((self.n_class, self.n_uniq_cols2))
        ci_class1 = np.zeros((self.n_class, self.n_uniq_cols2))
        m = np.zeros((self.n_class, self.n_uniq_cols2))
        observed_accuracy = np.zeros((self.n_class, self.n_uniq_cols2))
        z = cp.Variable((n, self.n_class))
        sel = cp.Parameter(n * self.n_class)                                

        lb_obj = cp.Minimize(sel.T @ cp.reshape(z, (n * self.n_class, 1),
            order='C'))
        ub_obj = cp.Maximize(sel.T @ cp.reshape(z, (n * self.n_class, 1),
            order='C'))

        # form primal problem's constraints
        constrs = [z >= 0, cp.sum(z, axis=1) == 1]

        for j in range(p):
            conf_mat_tr = cp.trace(z.T @ L_aug[j])
            constrs += [conf_mat_tr >= self.param_cts[1, j]-self.param_eps[j],
                    conf_mat_tr <= self.param_cts[1, j]+self.param_eps[j]]

        # form problems
        lb_prog = cp.Problem(lb_obj, constrs)
        ub_prog = cp.Problem(ub_obj, constrs)

        if patterns == "weak":
            
            for t in range(self.n_uniq_cols2):
            
                pattern_pt_inds = np.array(self.idx_patterns[t])
                a = Y_p[pattern_pt_inds, 0]
                m[0, t] = np.mean(a)
                observed_accuracy[0, t] = np.mean(1 - tl[0, pattern_pt_inds])
        else:

            for t in range(self.n_uniq_cols2):
                for j in range(self.n_class):
                  # select appropriate elements for objective
                      pattern_pt_inds = np.array(self.idx_patterns[j, t])
                      a = Y_p[pattern_pt_inds, j]
                      m[j, t] = np.mean(a)
                      if j == 0:
                          observed_accuracy[j, t] = np.mean(1 - tl[0, pattern_pt_inds])
                      else:
                        observed_accuracy[j, t] = np.mean(tl[0, pattern_pt_inds])
                  


        for i, prob in enumerate([lb_prog, ub_prog]):
            for t in range(self.n_uniq_cols2):
                for j in range(self.n_class):
                    # select appropriate elements for objective
                    # if patterns == "pattern":# "prob":
                    #     pattern_pt_inds = np.array(self.idx_patterns[j, t])
                    #     tmp = np.zeros(n * self.n_class)
                    #     tmp[pattern_pt_inds * self.n_class + j] = 1
                    #     sel.value = tmp
                    #     # solve for lower, then upper bounds
                    #     prob.solve(solver='MOSEK')
                    #     if j == 0:
                    #         if len(self.idx_patterns[j, t][0]) > 0:
                    #             ci_class0[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t][0]), 0, 1)
                    #     else:
                    #         if len(self.idx_patterns[j, t][0]) > 0:
                    #             ci_class1[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t][0]), 0, 1)
                                
                    if patterns == "pattern":
                        pattern_pt_inds = np.array(self.idx_patterns[j, t])
                        tmp = np.zeros(n * self.n_class)
                        tmp[pattern_pt_inds * self.n_class + j] = 1
                        sel.value = tmp
                        # solve for lower, then upper bounds
                        prob.solve(solver='MOSEK')
                        if j == 0:
                            if len(self.idx_patterns[j, t]) > 0:
                                ci_class0[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t]), 0, 1)
                        else:
                            if len(self.idx_patterns[j, t]) > 0:
                                ci_class1[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t]), 0, 1)
                                
                    elif patterns == "calibration":
                        pattern_pt_inds = np.array(self.idx_patterns[j, t])
                        tmp = np.zeros(n * self.n_class)
                        tmp[pattern_pt_inds * self.n_class + j] = 1
                        sel.value = tmp
                        # solve for lower, then upper bounds
                        prob.solve(solver='MOSEK')
                        if t == 0:
                            if len(self.idx_patterns[j, t][0]) > 0:
                                ci_class0[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t][0]), 0, 1)
                        else:
                            if len(self.idx_patterns[j, t]) > 0:
                                ci_class1[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t]), 0, 1)
                    elif patterns == "regions":
                        pattern_pt_inds = np.array(self.idx_patterns[j, t])
                        tmp = np.zeros(n * self.n_class)
                        tmp[pattern_pt_inds * self.n_class + j] = 1
                        sel.value = tmp
                        # solve for lower, then upper bounds
                        prob.solve(solver='MOSEK')
                        if j == 0:
                            ci_class0[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t]), 0, 1)
                        else:
                            ci_class1[i, t] = np.clip(prob.value/ len(self.idx_patterns[j, t]), 0, 1)
                    elif patterns == "weak":
                          pattern_pt_inds = np.array(self.idx_patterns[t])
                          tmp = np.zeros(n * self.n_class)
                          tmp[pattern_pt_inds * self.n_class + j] = 1
                          sel.value = tmp
                          # solve for lower, then upper bounds
                          prob.solve(solver='MOSEK')
                          if j == 0:
                              ci_class0[i, t] = np.clip(prob.value/ len(self.idx_patterns[t]), 0, 1)
                          else:
                              ci_class1[i, t] = np.clip(prob.value/ len(self.idx_patterns[t]), 0, 1)
        
        
        
        return ci_class0, ci_class1, m, observed_accuracy
    
    
    def get_confidence_prob(self, data):
        # L = check_weak_labels(data)#, n_weaks=n_weaks, random_guess=random_guess)
        L = data[0]
        Y_p = self.predict_proba(data)
        L_aug = self._initialize_L_aug(L)
        p, n, _ = L_aug.shape
        
    
        prob_values = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        r = len(prob_values)
        
        c = np.zeros((n, r*self.n_class))
        count = 0
        n_c = np.zeros((1, r*self.n_class))
        m = np.zeros((1, r*self.n_class))
        prob_perc = np.zeros((1, r*self.n_class))
        for j in range(r):
            for i in range(self.n_class):
                perc = np.percentile(Y_p[:, i], prob_values[j])
                idx = np.where(Y_p[:, i] >= perc)
                a = Y_p[idx, i]
                m[0, count] = np.mean(a)
                prob_perc[0, count] = perc
                aux = np.zeros((n, 1))
                y = np.zeros((self.n_class, 1))
                y[i] = 1
                aux[idx, 0] = 1
                c[:, count] = aux[:, 0]
                n_c[0, count] = len(idx[0])
                count += 1
            

        ci = np.zeros((2, r * self.n_class))
        z = cp.Variable((n, 1))
        sel = cp.Parameter(n)

        lb_obj = cp.Minimize((sel.T @ z))
        ub_obj = cp.Maximize((sel.T @ z))

        # form primal problem's constraints
        constrs = [z >= 0, z<=1]

        for j in range(p):
            conf_mat_tr = (1/(n))*(L_aug[j, :, 1]@z[:]+L_aug[j, :, 0]@(1 - z[:]))
            constrs += [conf_mat_tr >= self.param_probs[0, j],
                        conf_mat_tr <= self.param_probs[2, j]]
          
        # form problems
        lb_prog = cp.Problem(lb_obj, constrs)
        ub_prog = cp.Problem(ub_obj, constrs)
        
        for i, prob in enumerate([lb_prog, ub_prog]):
            count = 0
            for t in range(r):
                for j in range(self.n_class):
                    
                    sel.value = c[:, count]/n_c[0, count]
                    prob.solve(solver='MOSEK', warm_start=True)
                    ci[i, count] = prob.value
                    count += 1

        return ci, m, prob_perc
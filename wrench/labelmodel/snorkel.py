import logging
from typing import Any, Optional, Union

import cvxpy as cp
import numpy as np
import scipy as sp
from snorkel.labeling.model import LabelModel

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class Snorkel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 0.01,
                 l2: Optional[float] = 0.0,
                 n_epochs: Optional[int] = 100,
                 seed: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr'      : lr,
            'l2'      : l2,
            'n_epochs': n_epochs,
            'seed'    : seed or np.random.randint(1e6),
        }
        self.model = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False,
            weak: Optional[int] = None,
            n_weaks: Optional[int] = None,
            seed: Optional[int] = None,
            random_guess: Optional[int] = None,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        # L = check_weak_labels(dataset_train, n_weaks=n_weaks, random_guess=random_guess)
        # np.random.seed(seed)
        # np.random.shuffle(L.T)
        # L = L[:, 0:n_weaks]
        # n, m = L.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L = np.concatenate((L, r), axis = 1)
        L = dataset_train[0]
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        n_class = len(balance)
        self.n_class = n_class

        label_model = LabelModel(cardinality=n_class, verbose=verbose)
        label_model.fit(
            L_train=L,
            class_balance=balance,
            n_epochs=self.hyperparas['n_epochs'],
            lr=self.hyperparas['lr'],
            l2=self.hyperparas['l2'],
            seed=self.hyperparas['seed']
        )

        self.model = label_model
    def _initialize_L_aug(self, L):
        L = L.T
    
        L_aug = (np.arange(self.n_class) == L[..., None]).astype(int)
        
        # p, n = L.shape
        # L_aug = (np.arange(self.n_class) == L[..., None]).astype(float)
        # for i in range(p):
        #     for j in range(n):
        #         L_aug[i, j, 0] = 1-L[i, j]
        #         L_aug[i, j, 1] = L[i, j]
        return L_aug

           
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

        
    def get_confidence(self, data, patterns, n_weak_disagree):
        # L = check_weak_labels(data)#, n_weaks=n_weaks, random_guess=random_guess)
        # L = check_weak_labels(data)
        L = data[0]
        tl = data[1]

        self._get_pattern_LF(L, n_weak_disagree)
        Y_p = self.predict_proba(data)
        
        L_aug = self._initialize_L_aug(L)
        p, n, _ = L_aug.shape
        
        m = np.zeros((self.n_class, self.n_uniq_cols2))
        observed_accuracy = np.zeros((self.n_class, self.n_uniq_cols2))
    

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
                  
        return m

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], weak: Optional[int] = None, n_weaks: Optional[int] = None, random_guess: Optional[int] = None, seed: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        # L = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        L = dataset[0]
        # np.random.seed(seed)
        # np.random.shuffle(L.T)
        # L = L[:, 0:n_weaks]
        # n, m = L.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L = np.concatenate((L, r), axis = 1)
        return self.model.predict_proba(L)

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:44:32 2022

@author: Claudia Ghinato
"""

#TESTING

import pytest
import hypothesis
import numpy as np
#import random 

from complex.py import triupper

#%%

def test_1():
    prob= 0.8
    dim = 9
    matrix= np.zeros((dim,dim))
    triupper(prob,matrix)
    assert matrix[9][9]==0
    
    
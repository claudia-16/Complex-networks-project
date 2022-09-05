# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:44:32 2022

@author: Claudia Ghinato
"""

#TESTING

import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import random 

#from complex.py import triupper

#%%

# function for obtaining an upper triangular matrix (0s on the main diagonal)
# the function overwrite the input matrix
def triupper(prob, matrix):
    row= matrix.shape[0] # number of rows 
    col=matrix.shape[1]  # number of columns
    for i in range(0,row):
        for j in range(0,col):
            if (i>j or i==j):
                matrix[i][j]=0
            else:
                if (random.uniform(0,1)> 1-prob): # probability of having a link is greater than not having it
                    matrix[i][j]=1
                else:
                    matrix[i][j]=0

def test_diag_val():
    prob= 0.8
    dim = 9
    matrix= np.zeros((dim,dim))
    triupper(prob,matrix)
    assert matrix[dim-1][dim-1]==0
    
    
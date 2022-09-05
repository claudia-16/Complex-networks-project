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

### POSITIVE TESTS

# Below: test for verifying that each element on or below the diagonal is 0, whatever (or almost) is the size of the matrix 
@given(dim=st.integers(min_value=0, max_value=100))
def test_below_diag_val(dim):
    prob=0.8 #arbitrary value in [0,1]
    matrix= np.ones((dim,dim))
    triupper(prob, matrix)
    for i in range(0,dim):  # column index
        for j in range(i,dim-1): # row index
            assert matrix[j][i] ==0

# Below: test for veryfying that values above the main diagonal are actually inserted
@given(dim_bis=st.integers(min_value=0, max_value=100))
def test_above_diag_val(dim_bis):
    prob=1
    matrix= np.zeros((dim_bis,dim_bis))
    triupper(prob, matrix)
    assert np.sum(matrix)== (pow(dim_bis,2)-dim_bis)/2 


### NEGATIVE TESTS



### PROPERTY TEST
# DA SISTEMARE!!
@given(dim_tris=st.integers(min_value=0, max_value=100))
def test_same_dim(dim_tris):
    prob=0.8
    matrix_control= np.ones((dim_tris,dim_tris))
    matrix= np.ones((dim_tris,dim_tris))
    triupper(prob, matrix)
    #assert np.array(np.shape(matrix))== np.array([dim_tris,dim_tris])
    assert np.array(np.shape(matrix))== np.array(np.shape(matrix_control))



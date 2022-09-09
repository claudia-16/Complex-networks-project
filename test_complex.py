# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:44:32 2022

@author: Claudia Ghinato
"""

#TESTING

import random
import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
#import ipdb
#from grappa import should, expect

np.random.seed(18)

#%%

def triupper(prob, matrix):
    '''
    Function for obtaining an upper triangular matrix (0s on the main diagonal).
    The values above the main diagonal are randomly 0 or 1, according to the probability "prob".
    The function overwrites the input matrix.

    Parameters
    ----------
    prob : positive float
        Probability of having a link (value 1) in the upper triangular matrix
    matrix : numpy ndarray
        Input square matrix to be transformed

    '''
    if prob<0:
        raise ValueError("Probability must be >=0")
    if prob>1:
        raise ValueError("Probability must be <=1")
    row= matrix.shape[0] # number of rows
    col=matrix.shape[1]  # number of columns
    if row!=col:
        raise IndexError("The input matrix must be a square matrix")
    if not isinstance(matrix, np.ndarray):
        raise AttributeError("The input matrix must be a numpy ndarray")
    if matrix.ndim!=2:
        raise IndexError("The input matrix must be a 2-dimensional numpy array")
    for i in range(0,row):
        for j in range(0,col):
            if (i>j or i==j):
                matrix[i][j]=0
            else:
                # probability of having a link is greater than not having it
                if random.uniform(0,1)> 1-prob:
                    matrix[i][j]=1
                else:
                    matrix[i][j]=0


#%%

# triupper TESTS

### POSITIVE TESTS

@given(dim=st.integers(min_value=0, max_value=100))
def test_below_diag_val(dim):
    '''
    Test verifying that each element on the main diagonal
    or below it is 0, considering different sizes of the matrix

    '''
    prob=0.8 #arbitrary value in [0,1]
    matrix= np.ones((dim,dim))
    triupper(prob, matrix)
    for i in range(0,dim):  # column index
        for j in range(i,dim-1): # row index
            assert matrix[j][i] ==0


@given(dim_bis=st.integers(min_value=0, max_value=100))
def test_above_diag_val(dim_bis):
    '''
    Test verifying that above the main diagonal 1 values are actually inserted

    '''
    prob=1
    matrix= np.zeros((dim_bis,dim_bis))
    triupper(prob, matrix)
    assert np.sum(matrix)== (pow(dim_bis,2)-dim_bis)/2



### NEGATIVE TESTS

@given(neg_prob=st.floats(min_value=-100, max_value=-0.1))
def test_neg_prob(neg_prob):
    '''
    Test verifying an error is raised when the probability has a negative value

    '''
    matrix=np.ones((3,3))
    with pytest.raises(ValueError):
        triupper(neg_prob,matrix)


def test_sq_matrix():
    '''
    Test verifying an error is raised when the input matrix is not a square matrix

    '''
    prob=0.8
    matrix=np.array([[1,2,3,4],
                    [5,6,7,8]])
    with pytest.raises(IndexError):
        triupper(prob,matrix)


# The input matrix must be a numpy ndarray
def test_input_type():
    '''
    Test verifying an error is raised when the input matrix is not a numpy 2-dimansional array

    '''
    prob=0.8
    matrix_tuple=(1,2,3,4)
    matrix_list=[1,2,3,4,5]
    matrix_list_list=[[1,2,3],[4,5,6],[7,8,9]]
    matrix_dic = { "brand": "Ford", "model": "Mustang", "year": 1964}
    matrix_n= 2.5
    matrix_3darr= np.array([[[1,2,3],[4,5,6], [7,8,9]],
                              [[10,11,12],[13,14,15],[16,17,18]],
                              [[19,20,21],[22,23,24],[25,26,27]]])
    with pytest.raises(AttributeError):
        triupper(prob,matrix_tuple)
        triupper(prob, matrix_list)
        triupper(prob, matrix_list_list)
        triupper(prob,matrix_dic)
        triupper(prob, matrix_n)
    with pytest.raises(IndexError):
        triupper(prob,matrix_3darr)


def test_no_rep():
    '''
    Test verifying that the same input can (and in general will) provide different outputs

    Returns
    -------
    None.

    '''
    dim=5
    prob=0.5
    matrix_1=np.zeros((dim, dim))
    matrix_2=np.zeros((dim, dim))
    triupper(prob, matrix_1)
    triupper(prob, matrix_2)
    assert np.all(matrix_1== matrix_2) == False
    


### PROPERTY TESTS

@given(dim_tris=st.integers(min_value=0, max_value=100))
def test_same_dim(dim_tris):
    '''
    Test verifying the function does not change the size of the matrix

    '''
    prob=0.8
    matrix= np.ones((dim_tris,dim_tris))
    triupper(prob, matrix)
    assert np.size(matrix,axis=0)==dim_tris
    assert np.size(matrix,axis=1)==dim_tris



def test_same_type():
    '''
    Test verifying the output matrix is of the same type of the input matrix (numpy ndarray)

    '''
    prob=0.8
    matrix= np.ones((5,5))
    type_in= type(matrix)
    triupper(prob, matrix)
    type_out= type(matrix)
    assert type_out==type_in


#%%

def symm_block(dim,prob):
    '''
    The function produces a binary symmetric matrix having dimension given by the input value.
    The 1 values in the matrix are determined by the probability given in input.
    This function relies on triupper function.

    Parameters
    ----------
    dim : int
        Dimension of the matrix
    prob : float
        Probability that a matrix element has value 1

    Returns
    -------
    block : numpy array of int values

    '''
    aux=np.zeros((dim,dim), dtype=int)
    triupper(prob,aux)
    block=np.add(aux, np.transpose(aux))
    return block


#%%

# symm_block TESTS

### POSITIVE TESTS

@given(dim_4=st.integers(min_value=0, max_value=100))
def test_zero_diag(dim_4):
    '''
    Test verifying that each element on the main diagonal is 0,
    whatever is the size of the matrix.

    '''
    prob=0.8
    matrix=symm_block(dim_4,prob)
    assert (matrix[i][i]==0 for i in range(0,dim_4))


@given(prob=st.floats(min_value=0, max_value=1))
def test_symm(prob):
    '''
    Test verifying that the matrix generated is actually symmetric,
    whatever is the probability given in input.

    '''
    dim=10
    matrix=symm_block(dim, prob)
    assert (matrix[i][j]==matrix[j][i] for i,j in range(0,dim))
    
#!/usr/bin/env python
# coding: utf-8

# In[2]:


## module swap
''' swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].
    
    swapCols(v,i,j).
    Swaps columns of matrix [v].
    
    swapCramer(a, b, i).
    Swaps i-th column of matrix [a] with array [b].
'''
def swapRows(v,i,j):
    '''if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]
    return'''
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        temp = v[i].copy()
        v[i] = v[j]
        v[j] = temp

def swapCols(v,i,j):
    '''v[:,[i,j]] = v[:,[j,i]]
    return'''
    temp = v[:,j].copy()
    v[:,j] = v[:,i]
    v[:,i] = temp

def swapCramer(a, b, i):
    import numpy as np
    ai = a.copy()
    ai[:, i] = np.transpose(b)
    return ai


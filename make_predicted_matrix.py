import numpy as np
from numpy.random import *

#real_vec=rand(100)

def make_one_hot_vec(real_vec,theta):
    one_hot_vec=[0]*len(real_vec)
    for i, v in enumerate(real_vec):
        if v>theta:
            one_hot_vec[i]=1
        else: continue

    return (one_hot_vec)

#print(make_one_hot_vec(real_vec,0.2))




#real_vec=rand(10,10)

def make_one_hot_matrix(real_mat,theta):
    one_hot_matrix=np.zeros(real_mat.shape)
    for i,low in enumerate(real_mat):
        for j, col in enumerate(low):
            if col > theta: one_hot_matrix[i,j]=1
            else: continue
    return one_hot_matrix

#print (real_vec)
#print (make_one_hot_matrix(real_vec,0.1))

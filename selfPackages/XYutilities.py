#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
Some utilities for data generation and correlation matrix

"""

import numpy as np
import scipy as sc


def XY1dCouplings(BM,size=100,beta=1.0,num_states=10,PBC=False):
    
    # This is only the nearest neighbour interactions XY model on a 1D lattcie
    pottsCouplings = np.zeros([num_states,num_states])
    
    for i in np.arange(0,num_states):
        for j in np.arange(0,num_states):
            pottsCouplings[i,j] = np.cos((i-j)*2*np.pi/num_states)
        
    BM.layer.couplings[(np.eye(N=size,k=-1)+np.eye(N=size,k=+1)).astype(bool),:,:] = beta*pottsCouplings
    
    if PBC is True:
        BM.layer.couplings[0,-1] = beta*pottsCouplings
        BM.layer.couplings[-1,0] = beta*pottsCouplings
    
    return BM.layer.couplings

def XY2dCouplings(BM,size=20,beta1=1.0,beta2=1.0,num_states=10,PBC=False):
    
    # This is only the nearest neighbour XY model on a 2D square lattice
    # Calculate the KT transition beta
        
    pottsCouplings = np.zeros([num_states,num_states])
    N = size
    
    for i in np.arange(0,num_states):
        for j in np.arange(0,num_states):
            pottsCouplings[i,j] = np.cos((i-j)*2*np.pi/num_states)
            
    BM.layer.couplings[(np.eye(N=size**2,k=1) + np.eye(N=size**2,k=-1)).astype(bool),:,:] =  beta1*pottsCouplings
    BM.layer.couplings[(np.eye(N=size**2,k=size) + np.eye(N=size**2,k=-size)).astype(bool),:,:] =  beta2*pottsCouplings
    
    # Remove the couplings of the left most coloumn to that of the last coloumn in its previous row:
    for i in np.arange(0,N*N,N):
        BM.layer.couplings[i,i-1] = 0
        BM.layer.couplings[i-1,i] = 0
    
    if PBC is True:
        # Add the couplings of the first row to that of the last row and vice versa:
        for i in np.arange(0,N):
            BM.layer.couplings[i,(size**2)-N+i] = beta2*pottsCouplings
            BM.layer.couplings[(size**2)-N+i,i] = beta2*pottsCouplings
        # Add the couplings of the left most column to the right most coloumn:
            BM.layer.couplings[i,i+N-1] = beta1*pottsCouplings
            BM.layer.couplings[i+N-1,i] = beta1*pottsCouplings
    
    return BM.layer.couplings


def cosfit(x,peak,phase,freq):
    return peak*np.cos(x*freq + phase)

def topEV(data,method='Covariance'):
    # Top Eignvectors of the C_ij matrix:
    
    if method is 'Covariance':
        #Find the covariance matrix:
        covMAT = np.cov(data)
    elif method is 'Correlation':
        covMAT = np.corrcoef(data)
    Eval,Evec = np.linalg.eig(covMAT)
    return Evec[:,np.argmax(Eval)],np.max(Eval)

def corXY(data,num_configs,num_sites,method='Correlation'):
    
    cov_mat = np.zeros([num_configs*num_sites,num_configs*num_sites])
    
    if method is 'Covariance':
        for i1 in np.arange((num_sites)):
            for j1 in np.arange((num_configs)):
                for i2 in np.arange((num_sites)):
                    for j2 in np.arange((num_configs)):
                        cov_mat[i1*num_configs+j1, i2*num_configs+j2] = \
                        np.float(np.count_nonzero(np.logical_and(data[:,i1]==j1,data[:,i2]==j2)))/(len(data)) \
                        - np.float(np.count_nonzero(data[:,i1]==j1)*np.count_nonzero(data[:,i2]==j2))/(len(data)**2)

    if method is 'Correlation':
        
        diag_mat = np.zeros(num_configs*num_sites)
        
        for i1 in np.arange((num_sites)):
            for j1 in np.arange((num_configs)):
                        diag_mat[i1*num_configs+j1] = \
                        np.float(np.count_nonzero(np.logical_and(data[:,i1]==j1,data[:,i1]==j1)))/(len(data)) \
                        - np.float(np.count_nonzero(data[:,i1]==j1)*np.count_nonzero(data[:,i1]==j1))/(len(data)**2)
        
        for i1 in np.arange((num_sites)):
            for j1 in np.arange((num_configs)):
                for i2 in np.arange((num_sites)):
                    for j2 in np.arange((num_configs)):
                        
                        if np.sqrt(diag_mat[i1*num_configs+j1]*diag_mat[i2*num_configs+j2]) !=0:
                            cov_mat[i1*num_configs+j1, i2*num_configs+j2] = \
                            (np.float(np.count_nonzero(np.logical_and(data[:,i1]==j1,data[:,i2]==j2)))/(len(data)) \
                            - np.float(np.count_nonzero(data[:,i1]==j1)*np.count_nonzero(data[:,i2]==j2))/(len(data)**2)\
                            )/np.sqrt(diag_mat[i1*num_configs+j1]*diag_mat[i2*num_configs+j2])
                            
                        else:
                            cov_mat[i1*num_configs+j1, i2*num_configs+j2] = 0
    
    Eval,Evec = np.linalg.eig(cov_mat)
    
    return cov_mat,Evec,Eval
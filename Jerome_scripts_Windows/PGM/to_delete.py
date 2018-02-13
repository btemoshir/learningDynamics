#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:11:07 2017

@author: jerometubiana
"""

import numpy as np
import matplotlib.pyplot as plt
import rbm
import pgm
import layer
import bm
import utilities
reload(rbm)
reload(bm)
reload(pgm)
reload(layer)
reload(utilities)


B = 1000
N = 10

data = np.random.randint(0,high=2,size=[B,N])

weights = 1.0 * np.ones(B)
weights[ data[:,0] == 0] = 1e-8

weights[ data[:,3] != data[:,4] ] = 1e-8

BM = bm.BM(N=N)
BM.fit(data,weights=weights)


RBM = rbm.RBM(n_v = N,n_h = 1)
RBM.fit(data,weights=weights,batch_size=10,n_iter=10)


data = np.random.randint(0,high=2,size=[B,N])
data[:,0] *=0
data[:,0] +=1
data[:,4] = data[:,3] 

RBM = rbm.RBM(n_v = N,n_h = 1)
RBM.fit(data,batch_size=10)

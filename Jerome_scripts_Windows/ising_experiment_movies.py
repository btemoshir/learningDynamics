#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:16:39 2017

@author: jerometubiana
"""

import sys
sys.path.append('/Users/jerometubiana/Desktop/PhD/Ising_1d')
sys.path.append('/Users/jerometubiana/Desktop/PhD/rbm/PGM/')
sys.path.append('/Users/jerometubiana/Desktop/PhD/')
import numpy as np
import rbm
import layer
import matplotlib.pyplot as plt
reload(rbm)
reload(layer)

from ising import IsingModel
import estimate_correlations_ising
from sklearn.utils import shuffle
import weight_movie
reload(weight_movie)


def gen_Ising_data(beta,Nspins,Ndata):
    net = IsingModel(N=Nspins)
    net.set_beta ( beta )
    zeroone = False
    # First, estimate the correlation time to determine parameters.
    data = net.gen_data(0,int(1e4),1,zeroone=False,shuffled = False)
    time,error = estimate_correlations_ising.estimate_time_correlation(data,n_max = 1000)
    Nsteps = max(int(time * 5), 20)
    Nthermalize = Nsteps * 10
    print 'Ising beta : %s Nsteps: %s'%(beta,Nsteps)
    data = shuffle(net.gen_data(Nthermalize,Ndata,Nsteps,zeroone=zeroone ))
    pickle.dump({'data':data,'net':net,'Nspins':Nspins,'NData':Ndata}, 'data/Ising1D_N_%s_beta_%s.spydata'%(Nspins,beta))
    return data


def copy_and_rotate(template,Nspins):
    copied  = np.zeros([Nspins,Nspins])
    for j in range(Nspins):
        copied[j] = template[ (j+np.arange(Nspins))%Nspins ]
    return copied

#%%

Ndata = int(1e4)
Nspins = 300
for beta in [1,1.5,2]:
    print 'beta: %s'%beta
    data = gen_Ising_data(beta,Nspins,Ndata)
    print 'data generated'    
    for learning_rate in [0.001,0.01,0.05,0.1]:
        RBM = rbm.RBM(n_v = Nspins,n_h =1,visible='Spin',hidden='Spin',zero_field = True)
        result = RBM.fit(data,learning_rate = learning_rate,record=['W'],n_iter = 200,record_interval=40,lr_decay = False,batch_size = 10,N_MC =20,shuffle_data=True)
        print 'fit done'    
        weight_movie.gen_weights_movie(np.array(result['W']),'movie','experiment_beta_%s_Nspins_%s_Ndata_%s_lr_%s'%(beta,Nspins,Ndata,learning_rate))
        print 'movie done'
        
        
#%%
Ndata = int(3e4)
Nspins = 300
for beta in [1,1.5,2]:
    saveloadenv.load_env('data/Ising1D_N_%s_beta_%s.spydata'%(Nspins,beta),globals())
    data_sym = np.zeros([Ndata,Nspins])
    for i in range( Ndata/300):
        template = data[i]
        data_sym[300*i:300*(i+1)] = copy_and_rotate(template,Nspins)
    for learning_rate in [0.001,0.01,0.05,0.1]:
        RBM = rbm.RBM(n_v = Nspins,n_h =1,visible='Spin',hidden='Spin',zero_field = True)
        result = RBM.fit(data_sym,learning_rate = learning_rate,record=['W'],n_iter = 200,record_interval=40,lr_decay = False,batch_size = 10,N_MC =20,shuffle_data=True)
        print 'fit done'    
        weight_movie.gen_weights_movie(np.array(result['W']),'movie','experiment_invariant_beta_%s_Nspins_%s_Ndata_%s_lr_%s'%(beta,Nspins,Ndata,learning_rate))
        print 'movie done'        
        
    
    


        
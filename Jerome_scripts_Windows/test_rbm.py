#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:27:14 2017

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
import pickle
reload(rbm)
reload(layer)

Ndata = int(1e4)
Nspins = 100
data = np.random.rand(Ndata,Nspins)
data = np.sign(data - 0.5)
sites = np.arange(1,Nspins+1)

for i in [10,20,30,40,50,60,70,80,90]:
    data[:Ndata/2,i] = data[:Ndata/2,0]

for i in [15,25,35,45,55,65,75,85,95]:
    data[:Ndata/2,i] = data[:Ndata/2,5]

RBM = rbm.RBM(n_v = Nspins,n_h=4,visible='Spin',hidden='Spin',zero_field = True)

result = RBM.fit(data,record=['W'],record_interval=1,shuffle_data = True)

weights = np.array(result['W'])

for i in range(10):
    plt.plot(sites,weights[i,:,:].T); plt.show()
    
    

#%%

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
    pickle.dump({'data':data,'net':net,'Nspins':Nspins,'NData':Ndata}, open('data/Ising1D_N_%s_beta_%s.spydata'%(Nspins,beta),'wb') )
    return data


Ndata = int(1e4)
Nspins = 300
for learning_rate in [0.001,0.01,0.05,0.1]:
    for beta in [0.5,0.75,1,1.25,1.5,1.75,2]:
        print 'beta: %s'%beta
        data = gen_Ising_data(beta,Nspins,Ndata)
        print 'data generated'
        RBM = rbm.RBM(n_v = Nspins,n_h =1,visible='Spin',hidden='Spin',zero_field = True)
        result = RBM.fit(data,learning_rate = learning_rate,record=['W'],n_iter = 30,record_interval=10,lr_decay = False,batch_size = 10,N_MC =20)
        print 'fit done'    
        weight_movie.gen_weights_movie(np.array(result['W']),'movie','experiment_beta_%s_Nspins_%s_Ndata_%s_lr_%s'%(beta,Nspins,Ndata,learning_rate))
        print 'movie done'

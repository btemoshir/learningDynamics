#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:42:01 2017

@author: jerometubiana
"""
import matplotlib.pyplot as plt
import numpy as np
import saveloadenv2 as saveloadenv


saveloadenv.load_env('Ising1D_weights_N_300_beta_1_Ndata_390000_lr_0.006_6.spydata',globals())

weights = W_b
#%%
n_time = weights.shape[0]
n_v= weights.shape[2]
sites = range(1,n_v+1)
maxi = np.max(weights)
mini  = np.min(weights)
ylim = [ mini - 0.1 * (maxi-mini), maxi + 0.1 * (maxi-mini)]
fig, ax = plt.subplots()
ax.set_ylim(ylim)
ax.set_xlim([1,n_v])
ax.set_xlabel('Sites')
ax.set_ylabel('Weights')

sites2 = (np.array(sites)+145)%300
ax.plot(weights[[0,100,500,1000,5000]][:,0,sites2].T,linewidth=2)
ax.set_title('Receptive field fomation')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(18)

plt.legend(['T = %s'%x for x in [0,100,500,1000,5000]],fontsize=14,frameon=False)
plt.savefig('bump_formation.png')


#%%

weights = -w

n_time = weights.shape[0]
n_v= weights.shape[2]
sites = range(1,n_v+1)
maxi = np.max(weights)
mini  = np.min(weights)
ylim = [ mini - 0.1 * (maxi-mini), maxi + 0.1 * (maxi-mini)]
fig, ax = plt.subplots()
ax.set_ylim(ylim)
ax.set_xlim([1,n_v])
ax.set_xlabel('Sites')
ax.set_ylabel('Weights')

sites2 = (np.array(sites)-50)%300
liste = [250,500,1000]
ax.plot(weights[liste][:,0,sites2].T,linewidth=2)
ax.set_title('Receptive field diffusion')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(18)

plt.legend(['T = %s'%(x*60) for x in liste],fontsize=14,frameon=False)
plt.savefig('bump_diffusion.png')

#%%


saveloadenv.load_env('Ising1D_weights_N_300_beta_1_Ndata_390000_lr_0.006_30.spydata',globals())



weights = W_b
#%%
n_time = weights.shape[0]
n_v= weights.shape[2]
sites = range(1,n_v+1)
maxi = np.max(weights)
mini  = np.min(weights)
ylim = [ mini - 0.1 * (maxi-mini), maxi + 0.1 * (maxi-mini)]
fig, ax = plt.subplots()
ax.set_ylim(ylim)
ax.set_xlim([1,n_v])
ax.set_xlabel('Sites')
ax.set_ylabel('Weights')

sites2 = (np.array(sites)+145)%300
ax.plot(weights[-1,:,sites2],linewidth=2)
ax.set_title('Receptive field fomation')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(18)

#plt.legend(['T = %s'%x for x in liste],fontsize=14,frameon=False)
plt.savefig('two_bumps.png')


#%%


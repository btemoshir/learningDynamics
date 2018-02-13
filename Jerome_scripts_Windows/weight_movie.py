#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:53:07 2017

@author: jerometubiana
"""


import os
import matplotlib.pyplot as plt
import numpy as np

def gen_weights_movie(weights,folder,experiment,avconv = 'avconv',ylim = None):
    n_time = weights.shape[0]
    n_v= weights.shape[2]
    sites = range(1,n_v+1)
    if ylim is None:
        maxi = np.max(weights)
        mini  = np.min(weights)
        ylim = [ mini - 0.1 * (maxi-mini), maxi + 0.1 * (maxi-mini)]
    fig, ax = plt.subplots()
    ax.set_ylim(ylim)
    ax.set_xlim([1,n_v])
    ax.set_xlabel('Sites')
    ax.set_ylabel('Weights')
    line, = ax.plot(sites,weights[0,:,:].T)
    for i in range(n_time):
        line.set_ydata(weights[i,:,:].T)
        ax.set_title('Time steps: %s'%i)        
        plt.savefig(folder+ '/' + experiment+ 'tmp_%08d.png'%i)
    os.system(avconv + ' -y -i ' + folder + '/'+  experiment + 'tmp_%08d.png ' + folder + '/' + 'movie_weights_%s.mp4'%experiment)
    os.system('for f in '+folder+'/'+ experiment+'*tmp*'+'; do rm "$f"; done')
    return 'done'


"""
Created on Mon Mar 13 12:16:39 2017

@author: jerometubiana
"""

import sys
sys.path.append('/Users/jerometubiana/Desktop/PhD/Ising_1d')
sys.path.append('/Users/jerometubiana/Desktop/PhD/rbm/PGM/')
sys.path.append('/Users/jerometubiana/Desktop/PhD/')
sys.path.append('/users/tubiana/Ising_1d')
sys.path.append('/users/tubiana/PGM/')
sys.path.append('/users/tubiana/')
import numpy as np
import rbm
import layer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
reload(rbm)
reload(layer)
from functools import partial

from ising import IsingModel
#import estimate_correlations_ising
from sklearn.utils import shuffle
#import weight_movie
#reload(weight_movie)
from multiprocessing import Pool
import itertools


def track_particle(w):
    sites = np.arange(300)
    com = (np.abs(w)**2 * sites).sum(-1)/(np.abs(w)**2).sum(-1)
    width = np.sqrt( (np.abs(w)**2 * sites**2).sum(-1)/(np.abs(w)**2).sum(-1) - com**2)
    return com, width
    
def plot_TS(com,width, record_interval, experiment,folder=''):
    xaxis = np.arange(com.shape[0]) * record_interval
    fig,ax = plt.subplots()
    for i in range(com.shape[1]):
        plt.plot(xaxis, com[:,i])
        plt.fill_between(xaxis, com[:,i]- width[:,i]/2, com[:,i] + width[:,i]/2,
                         alpha=0.5, color='k')
    ax.set_xlabel('# mini-batch updates')
    ax.set_ylabel('Position')
    ax.set_title('Receptive field diffusion')
    ax.set_xlim([0,xaxis.max()])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    plt.savefig(folder+ 'plots/'+'diffusion_'+experiment + '.png')
    
    




def gen_Ising_data(beta,Nspins,Ndata,circular=None,folder =''):
    net = IsingModel(N=Nspins)
    net.set_beta ( beta )
    zeroone = False
    # First, estimate the correlation time to determine parameters.
    data = net.gen_data(0,int(1e4),1,zeroone=False,shuffled = False)
    time,error = estimate_correlations_ising.estimate_time_correlation(data,n_max = 1000)
    Nsteps = max(int(time * 5), 20)
    Nthermalize = Nsteps * 10
    print 'Ising beta : %s Nsteps: %s'%(beta,Nsteps)
    if circular is None:
        data = shuffle(net.gen_data(Nthermalize,Ndata,Nsteps,zeroone=zeroone ))
        pickle.dump({'data':data,'net':net,'Nspins':Nspins,'NData':Ndata}, open(folder+'data/Ising1D_N_%s_beta_%s_Ndata_%s.spydata'%(Nspins,beta,Ndata),'wb') )
    else:
        template = shuffle(net.gen_data(Nthermalize,Ndata/circular,Nsteps,zeroone=zeroone ))
        data = np.zeros([Ndata,Nspins])
        for i in range(Ndata/circular):
            data[i*circular:(i+1)*circular] = copy_and_rotate(template[i],Nspins)[::Nspins/circular]
        data = shuffle(data)
        pickle.dump({'data':data,'net':net,'Nspins':Nspins,'NData':Ndata}, open(folder+'data/Ising1D_N_%s_beta_%s_Ndata_%s_circular_%s.spydata'%(Nspins,beta,Ndata,circular),'wb') )
    
    return data


def copy_and_rotate(template,Nspins):
    copied  = np.zeros([Nspins,Nspins])
    for j in range(Nspins):
        copied[j] = template[ (j+np.arange(Nspins))%Nspins ]
    return copied



def train_and_movie(beta,Nspins,Ndata,circular=None,folder='',n_hiddens=1,learning_rate=1e-2, avconv = 'avconv', n_iter_ref = 200, Ndata_ref=int(3e4),record_interval=60,num_exp=0):
    if circular is None:
        experiment = 'N_%s_beta_%s_Ndata_%s'%(Nspins,beta,Ndata)
    else:
        experiment = 'N_%s_beta_%s_Ndata_%s_circular_%s'%(Nspins,beta,Ndata,circular)
    
    res = pickle.load(open(folder+'data/Ising1D_'+experiment+'.spydata','r'))
    data = res['data']
    res = {} # clean duplicate
    experiment += '_lr_%s'%learning_rate
    experiment += '_%s'%num_exp
    n_iter = n_iter_ref/(Ndata/Ndata_ref) #  6e5 updates, 1e4 frames
    RBM = rbm.RBM(n_v = Nspins,n_h =n_hiddens,visible='Spin',hidden='Spin',zero_field = True,random_state=num_exp)
    print experiment+'starting fit'
    result = RBM.fit(data,learning_rate = learning_rate,record=['W'],n_iter = n_iter,record_interval=record_interval,lr_decay = False,batch_size = 10,N_MC =200,shuffle_data=True,verbose=0)
    print experiment+' fit done'
    #weight_movie.gen_weights_movie(np.array(result['W']), folder +'movie/' ,'movie_'+experiment,avconv=avconv)
    #RBM = rbm.RBM(n_v = Nspins,n_h =n_hiddens,visible='Spin',hidden='Spin',zero_field = True,random_state=num_exp)
    #print experiment+'starting fit 2'
    #result_b = RBM.fit(data,learning_rate = learning_rate,record=['W'],n_iter = 1,record_interval=1,lr_decay = False,batch_size = 10,N_MC =20,shuffle_data=True,verbose=0)
    #print experiment+' fit done 2'
    data = []
    tmp= np.array(result['W'])
    com,width = track_particle(tmp)
    plot_TS(com,width,record_interval,'Ising1D_'+experiment,folder=folder)
    res = {'w':tmp}#,'W_b':np.array(result_b['W'])}
    print experiment+ ' movie done'
    pickle.dump(res, open(folder +'data/Ising1D_weights_' + experiment + '.spydata','wb'))
        
        

#%%

prod = True
if prod:
    folder = '/home/tubiana/Ising_1D/'
    Nspins = 300
    Ndatas = [int(3e4 * x) for x in range(1,21,3)]
    circulars = [None] + [3,10,50,100,300]
    learning_rates = [1e-3,3e-3,6e-3,1e-2,3e-2,6e-2]
    avconv = '/users/tubiana/libav-11.6/avconv'
    n_iter_ref = 1000
    Ndata_ref = Ndatas[0]
    record_interval = 100
else:
    folder = '/Users/jerometubiana/Desktop/PhD/Ising_1D/'
    Nspins = 20
    Ndatas = [int(2e3 * x) for x in range(1,4)]
    circulars = [None] + [2,10,20]
    learning_rates = [0.01,0.1]
    avconv = 'avconv'
    n_iter_ref = 3
    Ndata_ref = Ndatas[0]
    record_interval = 60
    
    
beta = 1

gen_data = False
if gen_data:
    pool = Pool()
    for Ndata in Ndatas:
        print Ndata
        f = partial(gen_Ising_data,beta,Nspins,Ndata,folder=folder)
        pool.map(f,circulars)

Ndatas = Ndatas[:-2]

train = False


def g(x):
    train_and_movie(beta,Nspins,x[2],circular=x[0],folder=folder,avconv=avconv,learning_rate=x[1],n_iter_ref = n_iter_ref,Ndata_ref=Ndata_ref,record_interval=record_interval)

if train:
    xs=itertools.product(circulars,learning_rates,Ndatas)
    pool = Pool(15)
#    f = partial(train_and_movie,beta,Nspins,folder=folder,avconv=avconv, n_iter_ref = n_iter_ref, Ndata_ref = Ndata_ref,record_interval = record_interval)
#    g = lambda x: f(x[0],circular=x[1],learning_rate=x[2])
    pool.map(g,xs)
    

#%%

def h(x):
    train_and_movie(beta,Nspins,390000,circular=None,folder=folder,avconv=avconv,learning_rate=0.01,n_iter_ref = n_iter_ref,Ndata_ref=Ndata_ref,record_interval=record_interval,num_exp=x)

def h2(x):
    train_and_movie(beta,Nspins,30000,circular=None,folder=folder,avconv=avconv,learning_rate=0.01,n_iter_ref = 2,Ndata_ref=Ndata_ref,record_interval=record_interval,num_exp=x,n_hiddens=2)

    
train_repeat = True

if train_repeat:
    xs = np.arange(21)
    pool = Pool(7)
    pool.map(h,xs)
    

def i(x):
    train_and_movie(beta,Nspins,390000,circular=None,folder=folder,avconv=avconv,learning_rate=0.01,n_iter_ref = n_iter_ref,Ndata_ref=Ndata_ref,record_interval=record_interval,num_exp=x,n_hiddens=2)
    
    
train_repeat2 = True

if train_repeat2:
    xs = np.arange(22,43)
    
    pool = Pool(7)
    pool.map(i,xs)




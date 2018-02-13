# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:17:08 2015

@author: jerometubiana
"""

#%% Script estimate temporal correlation.
from ising import *
from multiprocessing import Pool
from scipy.optimize import curve_fit
import pickle

def expo(x,a):
    return np.exp(-x/a)
    
    
def fast_autocorrel(arg):
    x = arg[0]
    n = arg[1]
    x_hat = np.fft.fft(x)
    return np.real(np.fft.ifft( np.abs(x_hat)**2 )[:n])/float(len(x)) - np.mean(x)**2



def spin_autocorr(data,n_max=10000):
#    pool = Pool(2)
#    result = np.mean(pool.map(fast_autocorrel,tuple([[data[:,k],n_max] for k in range(data.shape[1])]), chunksize= 10),axis=0)
#    pool.close()
#    pool.join()
    result = np.mean(map(fast_autocorrel,tuple([[data[:,k],n_max] for k in range(data.shape[1])])),axis=0)
    return result

def estimate_time_correlation(data,n_max = 10000):
    time_correlation = spin_autocorr(data, n_max = n_max)
    popt,pcov = curve_fit(expo,np.arange(n_max),time_correlation)
    return popt, pcov
     
#%%          
if __name__=='main':
    n_max = 100000
    net = IsingModel()
    tol_correl = 0.01
    
    time_correlation = []
    betas = np.arange(0.01,1.5,0.05) # -1/log(tanh(1.5)) = 23, 1/4 de la taille réelle...
    for beta in betas:
        print beta
        net.set_beta(beta)
        data = net.gen_data(0,int(1e6),1)
        time_correlation.append(spin_autocorr(data,n_max))
        
    time_correlation = np.array(time_correlation).transpose()    
        
    plt.plot(np.arange(n_max),time_correlation)
    plt.xlabel('number of MH iterations')
    plt.ylabel('pattern overlap <s_0 s_k >')
    plt.title('Pattern overlap evaluation')
    plt.legend(tuple([r'$\beta=%.1f$'%beta for beta in betas]))
    plt.savefig('plots/pattern_overlap_Ising1D.png')
    plt.close()
        
    time_corr = 0 * betas
    errors = 0 * betas
    for k in range(len(betas)):    
        popt,pcov = curve_fit(expo, np.arange(n_max), time_correlation[:,k])
        time_corr[k] = popt[0]
        errors[k] = pcov[0]
    
    plt.plot(betas,time_corr)
    plt.xlabel('beta')
    plt.ylabel('correlation time of Ising MCMC')
    plt.title('Metropolis Hasting typical time scale')
    plt.savefig('plots/correlation_time_Ising1D.png')
    plt.close()
    
    plt.plot(betas,errors)
    plt.xlabel('beta')
    plt.ylabel('fit error')
    plt.title('Fit error for Metropolis')
    plt.savefig('plots/correlation_time_Ising1D.png')
    plt.close()
    
    
    pickle.dump(globals(), open('time_correl_python1D.data','wb'))

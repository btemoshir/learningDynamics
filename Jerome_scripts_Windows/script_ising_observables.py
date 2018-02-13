# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:59:37 2015

@author: jerometubiana
"""

from ising import *
import pickle

N_spin = 256
betas = np.arange(0.1,3,0.1)

Ndata = int(1e6)
Nstep = 100
Nthermalize_init = int(1e6)
Nthermalize_next = int(1e4)

#Ndata = 1000
#Nstep = 1
#Nthermalize_init = int(1e4)
#Nthermalize_next = int(1e3)

models = ['1D','nD','nD','nD','nD']
extraparams_ = [[],[1],[2],[3]]

#models = ['1D']
#extraparams = [[]]

for model,extraparams in zip(models,extraparams_):

    net = IsingModel(N_spin,betas[0],model,extraparams)
    
    mean_energies = []
    var_energies = []
    Q_SK = []
    mean_magnets = []
    var_magnets = []
    correlation_strengths = []
    correlation_lengths = []
    
    for beta in betas:
        print beta
        net.set_beta(beta)
        if beta == betas[0]:
            mean_energy, mean_spin, mean_order_params, var_energy, covariance_spin, var_order_params = \
            net.simulate_observables(Nthermalize_init,Ndata,Nstep)        
            
        else:
            mean_energy, mean_spin, mean_order_params, var_energy, covariance_spin, var_order_params = \
            net.simulate_observables(Nthermalize_next,Ndata,Nstep)
            
        mean_energies.append(float(mean_energy))
        var_energies.append(float(var_energy))        
        Q_SK.append( float(np.sqrt(np.mean(mean_spin**2) - np.mean(mean_spin)**2 )))
        mean_magnets.append(float(mean_order_params))
        var_magnets.append(float(var_order_params))
        correlation_length,correlation_strength = correlation_length_strength(net,covariance_spin.copy())
        correlation_strengths.append(float(correlation_strength))
        correlation_lengths.append(float(correlation_length))

    pickle.dump(open('simulations_modele_%s_N%d_d%d.spydata'%(model,N_spin,net.d),'wb') )
    
#%%
    
#for model,extraparams in zip(models,extraparams):
#    load_env('simulations_modele_%s_N%d_d%d.spydata'%(model,N_spin,net.d),globals())
#    
#    plt.plot(betas,mean_energies,betas,[- math.tanh(beta) for beta in betas])
#    plt.title('Energie interne par spin modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/energie_interne_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    #
#    plt.plot(betas, var_energies, betas, [ (1- math.tanh(beta)**2) for beta in betas] ) #betas**2 * var_energies)
#    plt.title('Capacité thermique par spin modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/capacite_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    #
#    plt.plot(betas,Q_SK, betas, [0 for _ in betas])
#    plt.title('Parametre d\'ordre de Sherrington Kirkpatrick modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/qSK_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    #
#    plt.plot(betas, mean_magnets, betas, [0 for _ in betas])
#    plt.title('Magnétisation moyenne modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/magnetisation_moyenne_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    
#    #
#    plt.plot(betas, var_magnets, betas, [math.exp(2* beta) for beta in betas]) #betas * var_magnets)
#    plt.title('Susceptibilité moyenne par spin modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/susceptibilite_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    #
#    #
#    plt.plot(betas, correlation_strengths, betas,[1 for _ in correlation_strengths])
#    plt.title('Intensité de corrélation modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/intensite_correlation_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    #
#    plt.plot(betas, correlation_lengths, betas,[-1/math.log(math.tanh(beta)) for beta in betas])
#    plt.title('Longueur de corrélation modele Ising %s N=%d d=%d'%(model,N_spin,net.d) )
#    plt.savefig('plots/longueur_correlation_modele%s_N%d_d%d.png'%(model,N_spin,net.d),format='png')
#    plt.close()
#    
#    
#
#

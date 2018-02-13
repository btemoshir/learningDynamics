# -*- coding: utf-8 -*-
"""
Code pour la simulation du modèle d'Ising par Monte Carlo.
"""

import numpy as np
from scipy import sparse
#from scipy import weave
#import weave
import math
from sklearn.utils import shuffle
from scipy.optimize import curve_fit


def random_state(N,p):
    return ( 2*np.random.randint(0,high=2,size = [N,p]) - 1)
    
def index2coordinates(L,d,i):
    coordinates = np.zeros(d,dtype = int)
    for k in range(d):
        coordinates[k] = i%L
        i = (i - coordinates[k])/L
    return coordinates
    
def coordinates2index(L,d,coordinates):
    return int(sum([coordinates[k] * L**k for k in range(d)]))
    
def neighbours(L,d,i):
    coordinates = index2coordinates(L,d,i)
    neighbours = list()
    for v in range(d):
        neighbour = coordinates.copy()
        neighbour[v]= int( (neighbour[v] + 1)  % L )
        neighbours.append( coordinates2index(L,d,neighbour) )
        neighbour[v] = int( (neighbour[v] - 2)  % L )
        neighbours.append( coordinates2index(L,d,neighbour) )
    return neighbours
    
def get_distance(L,d,i,j):
    coo1 = index2coordinates(L,d,i)
    coo2 = index2coordinates(L,d,j)
    distance_sq = 0
    for i in range(d):
        distance_sq += min( abs(coo1[i] - coo2[i]), L - abs(coo1[i] - coo2[i]) )**2
    return math.sqrt(distance_sq)
    

def covariance_bining(net,covariance_spin):
    correlation = np.zeros(net.L/2)
    norm = np.zeros(net.L/2)
    for i in range(net.N):
        for j in range(i+1):
            index = math.floor(get_distance(net.L,net.d,i,j) )
            if index<=net.L/2-1:
                correlation[index]+= covariance_spin[i,j]
                norm[index]+=1
    return np.array([correlation[k]/float(norm[k]) if norm[k]<>0 else 0 for k in range(net.L/2)])
    

def expo_decay(x, a, b):
    return a * np.exp(-x/float(b)) + (1-a) * (x==0)


def correlation_length_strength(net,covariance_spin):
# Fit the distribution p_k = lam delta(k,0) + (1-lambda) alpha**k, with xi = -1/log(alpha)
# The estimators are: mu = sum(k p_k) / sum_{k>1} p_k = 1/(1-alpha) and sum_{k>1} p_k = lambda * alpha/(1-alpha)

    corr = covariance_bining(net,covariance_spin)
    r = np.arange(net.L/2)
    try:
        popt, pcov = curve_fit(expo_decay, r, corr,p0=[1,1])
    except RuntimeError:
        return 0.0,0.0
    
#    mass = np.sum(corr)
#    mu = np.sum(corr * np.arange(net.L/2)) / (mass - 1)
#    alpha = 1 - 1/mu
#    lam = (mass-1) * (1- alpha)/alpha
#        return -1/np.log(alpha), lam
    return popt[1],popt[0]
    


def low_rank_approximation(M,n):
    w, v = np.linalg.eigh(M)
    highest_eigs = abs(w).argsort()[-n:]
    bool_eigs = np.array([x in highest_eigs for x in range(M.shape[0])])

    return np.dot( v , np.dot(np.diag (bool_eigs * w ), v.transpose()) )

class IsingModel:
    
    def __init__(self,N=100,beta=1,model='1D',extraparams=[],zeroone = False):
        self.N = N
        self.beta = beta
        self.model = model
        self.state = random_state(N,1)
        self.fields = np.zeros(N)
        self.zeroone = zeroone
        
        if model=='1D':
            self.L = self.N # for compatibility with Ising nD
            self.d = 1 # for compatibility with Ising nD
            self.couplings = np.eye(N,k=1) + np.eye(N,k=-1)
            self.couplings[N-1,0]+=1
            self.couplings[0,N-1]+=1
            self.patterns = np.ones([N,1])
            
            self.neighbours=np.zeros([self.N,2*self.d],dtype=int) # for C++ loops.
            for k in range(self.N):
                self.neighbours[k,:] = neighbours(self.L,self.d,k)
                
            self.probas = np.array([np.exp( - 4 * max(k,0) * self.beta ) for k in range(-self.d, self.d +1 )])
            
        elif model=='MF':
            self.couplings = (np.ones([N,N]) - np.eye(N) ) / float(self.N)
            self.patterns = np.ones([N,1])
        
        elif model=='SK':
            self.couplings = np.random.normal(size=[N,N]) / float(math.sqrt(self.N))
            self.couplings = (self.couplings + self.couplings.transpose())/math.sqrt(2)
            for k in range(N):
                self.couplings[k,k] = 0
                
        elif model=='Hopfield':
            self.p = extraparams[0] if len(extraparams) else 1
            self.patterns = random_state(self.N,self.p)
            self.couplings = np.dot(self.patterns , self.patterns.transpose() ) / float(self.N)
            for k in range(self.N):
                self.couplings[k,k] = 0
                
        elif model == 'Custom':
            self.fields = np.zeros(N)
            self.couplings = np.zeros([self.N,self.N])
                
        elif model=='nD':        
            self.d = int(extraparams[0])
            self.L = int( N ** (1/float(self.d)) )
            self.N = self.L**self.d # make sure N is of the form L^d.

            if N <> self.N:
                print 'N is not power of d. New N: %s'%self.N
            self.state = random_state(self.N,1)

            # Construct the sparse matrix.
            I = [k for k in range(self.N) for _ in range(2*self.d) ]
            J = []
            self.neighbours=np.zeros([self.N,2*self.d],dtype=int)
            for k in range(self.N):
                nei = neighbours(self.L,self.d,k) 
                J+= nei
                self.neighbours[k,:] = nei
            K = [1/float(self.d) for k in range(self.N * 2 * self.d)]
            self.couplings = sparse.coo_matrix((K,(I,J)),shape = (self.N,self.N)).tocsr().todense()
            self.patterns = np.ones([self.N,1])
            
            self.probas = np.array([np.exp( - 4 * max(k,0) * self.beta ) for k in range(-self.d, self.d +1 )])
            
        else:
            print 'model %s not supported'%model
        
        self.energy = self.get_energy()[0,0]
        self.energy_square = self.energy**2
        self.order_params = self.get_order_params()[0,:] if not self.model in ['SK','Custom'] else []
        if not self.model in ['SK','Custom']:
            self.order_params_square = self.order_params**2
        

    
    def get_field(self,k):
        if self.model == 'nD':
            return np.array(self.couplings[k,:].dot(self.state) + self.fields[k])
        else:
            return np.dot(self.couplings[k,:] , self.state ) + self.fields[k]
        
    def get_energy(self):
        if self.model == 'nD':
            return - np.array(np.dot(self.state.T,self.fields)+ np.dot(self.state.transpose(), self.couplings.dot(self.state))) * 1/float(2 * self.N)
        else:
            return - (np.dot(self.state.T,self.fields)+ np.dot(self.state.transpose(),np.dot(self.couplings,self.state)) ) * 1/float(2 * self.N)
        
    def set_beta(self,beta):
        self.beta = beta
        if self.model in ['1D','nD']:
            self.probas = np.array([np.exp( - 4 * max(k,0) * self.beta ) for k in range(-self.d, self.d +1 )])
    
    def get_order_params(self):
        if self.model in ['1D','nD','MF','Hopfield']:
            return np.dot(self.state.transpose(),self.patterns) * 1/float(self.N)
        else:
            return 0
                        
    def MHstep(self,Nstep):
        for _ in range(Nstep):
            k = np.random.randint(0,self.N)
            if self.zeroone:
                deltaE = (2 * self.state[k]-1) * self.get_field(k)
            else:
                deltaE = 2 * self.state[k] * self.get_field(k) #  ( [- (-s_k) * h] - [- (s_k)*h] )
            try:
                p = math.exp(- deltaE * self.beta )
            except OverflowError:
                p = 0
            if np.random.random() < p:
                if self.zeroone:
                    self.state[k] = 1-self.state[k]
                else:
                    self.state[k]*= -1
                self.energy+= deltaE[0] /float(self.N)
                self.energy_square = self.energy**2
                
                if self.model in ['1D','nD','MF','Hopfield']:
                    self.order_params += 2 * self.patterns[k,:] * self.state[k] / float(self.N)
                    self.order_params_square = self.order_params**2
                    
                    
    #def CPPMHstep(self, Nstep):
    #    "The same Monte Carlo sampling in C++"
    #
    #    if self.model not in ['1D','nD']:
    #        print 'models not supported'
    #        return
    #    else:
    #        neighbour = self.neighbours
    #        deltaE = np.zeros(1,dtype=float)
    #        deltaM = np.zeros(1,dtype=float)
    #        S = self.state
    #        d = self.d
    #        N = self.N
    #        PW = self.probas
    #        
    #        
    #        code="""
    #        using namespace std;
    #        for (int itt=0; itt<Nstep; itt++){
    #            int i = static_cast<int>(drand48()*N);
    #            int WF = 0;
    #            for (int j=0; j<2*d; j++){
    #            WF+=S(neighbour(i,j)) ;
    #            }
    #            double P = PW( d+S(i)*WF/2 );
    #            if (P > drand48()){ // flip the spin
    #                S(i) *= -1;
    #                deltaE(0) += -2*S(i)*WF;
    #                deltaM(0) += 2*S(i) ;
    #            }
    #            }
    #            deltaE(0) *= float(1)/float(N) ;
    #            deltaM(0) *= float(1)/float(N) ;
    #        """
    #        weave.inline(code, ['neighbour', 'deltaE', 'deltaM','S','d','N','PW','Nstep'],
    #                     type_converters=weave.converters.blitz, compiler = 'gcc')
    #        
    #        self.energy += deltaE[0]
    #        self.order_params += deltaM[0]
    #        self.energy_square = self.energy**2
    #        self.order_params_square = self.order_params**2
                    
          
                    

    def simulate_observables(self,Nthermalize,Ndata,Nstep):
        mean_energy = 0
        var_energy = 0
        mean_order_params = 0 * self.order_params
        var_order_params = 0 * self.order_params
        covariance_spin = np.zeros([self.N,self.N])
        mean_spin = np.zeros([self.N,1])
#            if self.model in ['1D','nD']:
#                self.CPPMHstep(Nthermalize)
#            else:
        self.MHstep(Nthermalize)
            
        for _ in range(Ndata):
            mean_energy += self.energy
            var_energy += self.energy_square
            mean_order_params += self.order_params
            var_order_params += self.order_params_square
            mean_spin += self.state
            covariance_spin += np.dot(self.state,self.state.transpose())
#                if self.model in ['1D','nD']:
#                    self.CPPMHstep(Nstep)
#                else:
            self.MHstep(Nstep)
    
        mean_energy *= 1/float(Ndata)
        mean_order_params *= 1/float(Ndata)
        mean_spin *= 1/float(Ndata)
        var_energy = float(self.N) * (var_energy/float(Ndata) - mean_energy**2 ) # Rescale by N to measure variance of total energy per spin
        var_order_params = float(self.N) * (var_order_params/float(Ndata) - mean_order_params**2) # Rescale by N to measure variance of total magnetization per spin
        covariance_spin = covariance_spin/float(Ndata) - np.dot(mean_spin,mean_spin.transpose())
        return mean_energy, mean_spin, mean_order_params, var_energy, covariance_spin, var_order_params
        
        
    def gen_data(self,Nthermalize,Ndata,Nstep, zeroone=True,shuffled=True):
#        if self.model in ['1D','nD']:
#            self.CPPMHstep(Nthermalize)
#        else:
        self.MHstep(Nthermalize)
        data = [self.state.copy()]
        for _ in range(Ndata-1):
#            if self.model in ['1D','nD']:
#                self.CPPMHstep(Nstep)
#            else:
            self.MHstep(Nstep)
            data.append(self.state.copy())
        if zeroone:
            if shuffled:
                if self.zeroone:
                    return shuffle(np.array(data)[:,:,0])
                else:                    
                    return shuffle((np.array(data)[:,:,0] + 1)/2)
            else:
                if self.zeroone:
                    return np.array(data)[:,:,0]
                else:
                    return (np.array(data)[:,:,0] + 1)/2
        else:
            if shuffled:
                if self.zeroone:
                    return shuffle(2*np.array(data)[:,:,0]-1)
                else:
                    return shuffle(np.array(data)[:,:,0])
            else:
                if self.zeroone:
                    return 2*np.array(data)[:,:,0]-1
                else:
                    return np.array(data)[:,:,0]
            
    def compute_beta_energy(self,data):
        energy = []
        for k in range(data.shape[0]):
            self.state = np.atleast_2d(data[k,:]).transpose()
            energy.append(self.beta * np.concatenate(self.get_energy()))
        return np.concatenate(np.array(energy))
        
    def compute_expected_observables(self):
        if (self.model=='1D') or (self.model=='nD' and self.d ==1):
            mean_magnet = 0
            std_magnet = np.exp(self.beta)/np.sqrt(self.N)
            q_SK = 0
            correlation_length = -1/np.log(np.tanh(self.beta))
            correlation_strength = 1
            mean_beta_energy = - self.beta * np.tanh(self.beta)
            std_beta_energy = self.beta * np.sqrt( 1 - np.tanh(self.beta)**2)/np.sqrt(self.N)
        else:
            print 'not supported'
        return mean_magnet,std_magnet,q_SK,correlation_length,correlation_strength,mean_beta_energy,std_beta_energy
            
        
#%% Junk
        
#def distrib(lamb,xi,L):
#    d = np.exp(-np.arange(L)/float(xi))
#    d*=lamb
#    d[0]+=(1-lamb)
#    return d
#    
#        
#        
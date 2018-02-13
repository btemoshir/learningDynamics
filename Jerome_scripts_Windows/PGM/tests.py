# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 23:29:43 2016

@author: jerometubiana
"""

import harmomium
import time
RBM = harmonium.Harmonium(n_visibles = 1000, n_hiddens =1000)

datav = np.random.randint(0,high=2, size = [1000,1000])
t = time.time()
datav,datah = RBM.gibbs(datav)
print time.time() -t 



#%%

import layer
N = 100
Layer = layer.Layer(N = N, nature='Bernoulli')
Layer.fields = np.random.randn(N)
inputs = np.zeros([1,N])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter(Layer.fields, tmp); plt.show()
inputs2 = np.zeros([10000,N])
tmp2 = Layer.sample_from_inputs(inputs2).mean(0)
plt.scatter(tmp,tmp2); plt.show()

inputs3 = np.random.randn(100,N)
tmp3 = Layer.mean_from_inputs(inputs3)
plt.scatter(tmp3, 1/(1+ np.exp(-inputs3-Layer.fields) )); plt.show()

inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4); plt.show()

inputs5 = inputs4
tmp5 = Layer.mean2_from_inputs(inputs5)
plt.scatter(tmp4,tmp5); plt.show()

inputs6 = inputs4
tmp6 = Layer.var_from_inputs(inputs5)
plt.scatter(tmp6,tmp5*(1-tmp5)); plt.show()


#%%


import layer
reload(layer)
N = 100
Layer = layer.Layer(N = N, nature='Spin')
Layer.fields = np.random.randn(N)
inputs = np.zeros([1,N])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter(Layer.fields, tmp); plt.show()
inputs2 = np.zeros([10000,N])
tmp2 = Layer.sample_from_inputs(inputs2).mean(0)
plt.scatter(tmp,tmp2); plt.show()

inputs3 = np.random.randn(100,N)
tmp3 = Layer.mean_from_inputs(inputs3)
plt.scatter(tmp3, np.tanh(inputs3+Layer.fields) ); plt.show()

inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4)


inputs5 = inputs4
tmp5 = Layer.mean2_from_inputs(inputs5)
plt.scatter(tmp5,tmp5*0 +1); plt.show()

inputs6 = inputs4
tmp6 = Layer.var_from_inputs(inputs5)
plt.scatter(tmp6,1-tmp4**2); plt.show()



#%%
reload(layer)
N = 100
Layer = layer.Layer(N = N, nature='Gaussian')
Layer.b = np.random.randn(N)
Layer.a =np.ones(N) + 0.1 * np.random.randn(N)
inputs = np.zeros([1,N])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter(-Layer.b/Layer.a, tmp); plt.show()
inputs2 = np.zeros([10000,N])
tmp2 = Layer.sample_from_inputs(inputs2).mean(0)
plt.scatter(tmp,tmp2); plt.show()

inputs3 = np.random.randn(100,N)
tmp3 = Layer.mean_from_inputs(inputs3)
plt.scatter(tmp3, (inputs3-Layer.b )/Layer.a); plt.show()


inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4); plt.show()


inputs5 = inputs4
tmp5 = Layer.mean2_from_inputs(inputs5)
plt.scatter(tmp5,((inputs5 -Layer.b)/Layer.a)**2 + 1/Layer.a); plt.show()

inputs6 = inputs4
tmp6 = Layer.var_from_inputs(inputs5)
plt.scatter(tmp6,1/Layer.a + tmp6 *0); plt.show()

inputs7 = np.ones([10000,N])
tmp7 = (Layer.sample_from_inputs(inputs7)**2).mean(0)
tmp7b = Layer.mean2_from_inputs(inputs7).mean(0)
tmp8 = (Layer.sample_from_inputs(inputs7)).var(0)
tmp8b = Layer.var_from_inputs(inputs7).mean(0)

plt.scatter(tmp7,tmp7b);plt.show()
plt.scatter(tmp8,tmp8b);plt.show()

#%%

reload(layer)
N = 100
Layer = layer.Layer(N = N, nature='ReLU')
Layer.theta_plus= np.random.randn(N)
Layer.theta_minus = np.random.randn(N)
Layer.a =np.ones(N) + 0.1 * np.random.randn(N)
inputs = np.zeros([1,N])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter((Layer.theta_minus - Layer.theta_plus)/Layer.a, tmp); plt.show()
inputs2 = np.zeros([10000,N])
tmp2 = Layer.sample_from_inputs(inputs2).mean(0)
plt.scatter(tmp,tmp2); plt.show()

inputs3 = np.random.randn(100,N)
tmp3 = Layer.mean_from_inputs(inputs3)
#plt.scatter(tmp3, (inputs3-Layer.b )/Layer.a); plt.show()


inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4); plt.show()


inputs7 = np.ones([10000,N])
tmp7 = (Layer.sample_from_inputs(inputs7)**2).mean(0)
tmp7b = Layer.mean2_from_inputs(inputs7).mean(0)
tmp8 = (Layer.sample_from_inputs(inputs7)).var(0)
tmp8b = Layer.var_from_inputs(inputs7).mean(0)

plt.scatter(tmp7,tmp7b);plt.show()
plt.scatter(tmp8,tmp8b);plt.show()


#%%

reload(layer)
N = 100
Layer = layer.Layer(N = N, nature='ReLU+')
Layer.theta_plus= np.random.randn(N)
Layer.a =np.ones(N) + 0.1 * np.random.randn(N)
inputs = np.zeros([1,N])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter((- Layer.theta_plus)/Layer.a, tmp); plt.show()
inputs2 = np.zeros([10000,N])
tmp2 = Layer.sample_from_inputs(inputs2).mean(0)
plt.scatter(tmp,tmp2); plt.show()

inputs3 = np.random.randn(100,N)
tmp3 = Layer.mean_from_inputs(inputs3)
#plt.scatter(tmp3, (inputs3-Layer.b )/Layer.a); plt.show()


inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4); plt.show()

inputs7 = np.ones([10000,N])
tmp7 = (Layer.sample_from_inputs(inputs7)**2).mean(0)
tmp7b = Layer.mean2_from_inputs(inputs7).mean(0)
tmp8 = (Layer.sample_from_inputs(inputs7)).var(0)
tmp8b = Layer.var_from_inputs(inputs7).mean(0)

plt.scatter(tmp7,tmp7b);plt.show()
plt.scatter(tmp8,tmp8b);plt.show()

#%%


import layer
reload(layer)
N = 100
n_c = 5
Layer = layer.Layer(N = N, nature='Potts',n_c = n_c)
Layer.fields = np.random.randn(N,n_c)
inputs = np.zeros([1,N,n_c])
tmp = Layer.mean_from_inputs(inputs)
plt.scatter( np.exp(-Layer.fields)/np.exp(-Layer.fields).sum(-1)[:,np.newaxis] , tmp); plt.show()
inputs2 = np.zeros([10000,N,n_c])
tmp2 = Layer.sample_from_inputs(inputs2)
for i in range(n_c):
    plt.scatter(tmp[0,:,i],(tmp2 ==i).mean(0)); 
plt.show()

inputs3 = np.random.randn(100,N,n_c)
tmp3 = Layer.mean_from_inputs(inputs3)
plt.scatter(tmp3, np.exp(-inputs3-Layer.fields[np.newaxis,:,:])/(np.exp(-inputs3-Layer.fields[np.newaxis,:,:]) ).sum(-1)[:,:,np.newaxis]); plt.show()

inputs4 = inputs3
tmp4 = Layer.transform(inputs4)
plt.scatter(tmp3, tmp4)



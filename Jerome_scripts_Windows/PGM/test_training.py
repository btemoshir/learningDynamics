import rbm
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import MNIST_utils
reload(rbm)


mndata = MNIST(u'/Users/jerometubiana/Desktop/PhD/MNIST/data')
data_mnist,labels_mnist = mndata.load_training()
data_mnist_test,labels_mnist_test = mndata.load_testing()
data_mnist = np.array(data_mnist).reshape([60000, 28,28])[:,4:-4,4:-4].reshape([60000,400])
data_mnist_test = np.array(data_mnist_test).reshape([10000, 28,28])[:,4:-4,4:-4].reshape([10000,400])
data_mnist = np.array(map(lambda x: x>= 128,data_mnist))
data_mnist_test = np.array(map(lambda x: x>= 128,data_mnist_test))


#%%
reload(rbm)
RBM = rbm.RBM(n_v = 400,n_h = 100,hidden='Bernoulli')
RBM.fit(data_mnist,n_iter = 10)
image1 = MNIST_utils.show_weights(RBM,sort='beta')

#%%
 
data_mnist2 = np.array(data_mnist,dtype=int)
 #%%
reload(rbm)
RBM2 = rbm.RBM(n_v = 400, n_h =100, visible ='Potts',n_cv = 2)
RBM2.fit(data_mnist2,n_iter = 10)



RBM3 = rbm.RBM(n_v = 400,n_h = 100);
RBM3.weights = RBM2.weights[:,:,0] - RBM2.weights[:,:,1]
image2 = MNIST_utils.show_weights(RBM3,sort='beta')

#%%

reload(rbm)
RBM2 = rbm.RBM(n_v = 400, n_h =100, visible ='Potts',n_cv = 2)
RBM2.fit(data_mnist2,learning_rate = 1e-10,n_iter =1)


datav,datah = RBM2.run_MC(n_samples = 100,n_steps = 1)

plt.imshow(datav.mean(0).mean(0).reshape([20,20]))

#%%

data = np.random.randint(0,high=10, size =[10000,50])
data[data == 4] = 6
data[:,1] = data[:,0]
data[:,10] = data[:,0]
data[:,20] = 9 - data[:,15]

RBM4 =rbm.RBM(n_v = 50, n_h = 15, visible ='Potts',n_cv = 10)
RBM4.fit(data,learning_rate = 2,CD = True)


plt.imshow((RBM4.weights**2).sum(-1),interpolation='none',aspect='auto'); plt.colorbar()
plt.imshow(RBM4.weights[0,:,:],interpolation='none',aspect='auto'); plt.colorbar()

#%% Ca semble marcher, mais l'interpretation n'est pas claire...

reload(rbm)
data = np.random.randint(0,high=3, size = [10000,5])
data[:,1] = data[:,4]

RBM4 = rbm.RBM(n_v = 5, n_h =3, visible ='Potts', hidden = 'Bernoulli',n_cv = 3)


RBM4.fit(data,learning_rate = 0.5,n_iter =20,CD =True)


datav,datah = RBM4.run_MC()
datav = datav.reshape([datav.shape[0] * datav.shape[1],datav.shape[2]]);
import utilities
correl = utilities.average_product(data,data,c1=3,c2=3)
correlv = utilities.average_product(datav,datav,c1=3,c2=3)
                            
print correl[1,4]
print correlv[1,4]
                     
#%% Potts RBM

reload(rbm)
RBM5 = rbm.RBM(n_v = 400, n_h = 1, hidden = 'Potts',n_ch = 20)
RBM5.fit(data_mnist)

A = (RBM5.vlayer.fields[:,np.newaxis] + RBM5.weights[0])
for i in range(20):
    plt.imshow(A[:,i].reshape([20,20])); plt.colorbar(); plt.show()
    
B = RBM5.weights[0]
for i in range(20):
    plt.imshow(B[:,i].reshape([20,20])); plt.colorbar(); plt.show()
    
#%% Double Potts RBM
reload(rbm)
RBM6 = rbm.RBM(n_v = 400, n_h =1, visible = 'Potts', hidden = 'Potts', n_cv =2, n_ch = 20)
RBM6.fit(data_mnist2)



A = (RBM6.vlayer.fields[:,1,np.newaxis] - RBM6.vlayer.fields[:,0,np.newaxis] + RBM6.weights[0,:,:,1] - RBM6.weights[0,:,:,0])
for i in range(20):
    plt.imshow(A[:,i].reshape([20,20])); plt.colorbar(); plt.show()
    
#B = RBM5.weights[0]
#for i in range(20):
#    plt.imshow(B[:,i].reshape([20,20])); plt.colorbar(); plt.show()    


#%% Check seeds
reload(rbm)
RBM1 = rbm.RBM(random_state = 1)
RBM2 = rbm.RBM(random_state = 1)

datav1,datah1 = RBM1.run_MC()
datav2,datah2 = RBM2.run_MC()

RBM1.fit(datav1[0])
RBM2.fit(datav2[0])





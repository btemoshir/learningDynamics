
import numpy as np
import utilities
import sequence_logo
import sequence_logo_gaps
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os



aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
aadict = {aa[k]:k for k in range(len(aa))}

def seq2num(string):
    if type(string) == str:
        return np.array([aadict[x] for x in string])[np.newaxis,:]
    elif type(string) ==list:
        return np.array([[aadict[x] for x in string_] for string_ in string])


def num2seq(num):
    return [''.join([aa[x] for x in num_seq]) for num_seq in num]
    

def saveRBM(filename,RBM):
    pickle.dump(RBM,open(filename,'wb'))

def loadRBM(filename):
    return pickle.load(open(filename,'r'))

def get_mutual_vh(data,RBM):
	if not RBM.hidden in ['Bernoulli','Spin']:
		print 'mutual not supported'
		return None
	else:
	    epsilon = 1e-6
	    mu_data = RBM.mean_hiddens(data)
	    mu = mu_data.mean(0)
	    if RBM.hidden == 'Spin':
	    	mu = (1.+mu)/2.
	    H = -mu * np.log(mu+epsilon) - (1-mu) * np.log(1-mu+epsilon)
	    H_cond = (-mu_data * np.log(mu_data+epsilon) - (1-mu_data) * np.log(1-mu_data+epsilon)).mean(0)
	    I = H-H_cond
	    return I

def get_sparsity(W,a=3,include_gaps=False):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    tmp = np.sqrt((W_**2).sum(-1))
    p = ((tmp**a).sum(1))**2/(tmp**(2*a)).sum(1)
    return p/W.shape[1]

def get_beta(W,include_gaps=False):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    return  np.sqrt( (W_**2).sum(-1).sum(-1) )

def get_theta(RBM):
	if RBM.hidden == 'ReLU':
		return (RBM.hlayer.theta_plus+ RBM.hlayer.theta_minus)/(2*RBM.hlayer.a)
	elif RBM.hidden == 'ReLU+':
		return (RBM.hlayer.theta_plus - RBM.hlayer.b)/RBM.hlayer.a
	elif RBM.hidden =='dReLU':
		return (1-RBM.hlayer.eta**2)/2 * (RBM.hlayer.theta_plus + RBM.hlayer.theta_minus)/RBM.hlayer.a
	else:
		print 'get_theta not supported for hidden %s'%RBM.hidden

def get_beta_gaps(W):
    return np.sqrt( (W**2)[:,:,-1].sum(-1) )

def get_hidden_input(data,RBM,normed=False,offset=True):
    if normed:
        mu = utilities.average(data,c=21)
        norm_null = np.sqrt(  ((RBM.weights**2 * mu).sum(-1) - (RBM.weights*mu).sum(-1)**2).sum(-1) )
        return (RBM.vlayer.compute_output(data,RBM.weights) - RBM.hlayer.b[np.newaxis,:])/norm_null[np.newaxis,:]
    else:
        if offset:
            return (RBM.vlayer.compute_output(data,RBM.weights) - RBM.hlayer.b[np.newaxis,:])
        else:
            return (RBM.vlayer.compute_output(data,RBM.weights) )


def get_correlation(data,RBM,weights=None):
    projections = RBM.vlayer.compute_output(data,RBM.weights)
    transformed = RBM.hlayer.mean_from_inputs(projections)
    mu = utilities.average(data,c=21,weights=weights)
    mu_h = utilities.average(transformed,weights=weights)
    coactivation = utilities.average_product(transformed,data,c2=21,weights=weights)
    correlation = coactivation - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
    return correlation

def get_normed_conditional_mean(data,RBM,weights=None,with_cond_var = True):
    projections = RBM.vlayer.compute_output(data,RBM.weights)
    h = RBM.hlayer.mean_from_inputs(projections)
    var_h_cond = RBM.hlayer.var_from_inputs(projections)
    if not with_cond_var:
    	var_h_cond *= 0
    mu = utilities.average(data,c=21,weights=weights)
    mu_h = utilities.average(h,weights=weights)
    var_h = utilities.average(h**2,weights=weights) - utilities.average(h,weights=weights)**2 + utilities.average(var_h_cond,weights=weights)
    coactivation = utilities.average_product(h,data,c2=21,weights=weights)
    covariance = coactivation  - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
    delta = covariance/(var_h[:,np.newaxis,np.newaxis])
    return delta

def get_normed_conditional_mean_sparse(data,RBM,weights=None,with_cond_var = True,q=2):
    projections = RBM.vlayer.compute_output(data,RBM.weights)
    h = RBM.hlayer.mean_from_inputs(projections)
    var_muh = utilities.average(h**2,weights=weights) - utilities.average(h,weights=weights)**2
    var_h_cond = RBM.hlayer.var_from_inputs(projections)
    mu = utilities.average(data,c=21,weights=weights)
    mu_h = utilities.average(h,weights=weights)
    var_h =  var_muh + utilities.average(var_h_cond,weights=weights)
    coactivation = utilities.average_product(h,data,c2=21,weights=weights)
    covariance = coactivation  - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
    N = data.shape[0]
    null_covariance = np.sqrt( (N-1.0)/N**2 * (mu * (1-mu))[np.newaxis,:,:] * var_muh[:,np.newaxis,np.newaxis] )
    mask = np.abs(covariance) > q * null_covariance
    covariance *= mask
    delta_sparse = covariance/(var_h[:,np.newaxis,np.newaxis])
    return delta_sparse


def get_delta_conditional_mean(data,RBM,weights=None): # Pour Bernoulli
	if not RBM.hidden in ['Bernoulli','Spin']:
		print 'delta not supported'
		return None
	else:
	    h = RBM.mean_hiddens(data)
	    if RBM.hidden == 'Spin':
	    	h = (1.+h)/2.
	    mu = utilities.average(data,c=21,weights=weights)
	    mu_h = utilities.average(h,weights=weights)
	    coactivation = utilities.average_product(h,data,c2=21,weights=weights)
	    covariance = coactivation  - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
	    delta = covariance/((mu_h * (1-mu_h) )[:,np.newaxis,np.newaxis])
    	return delta

def get_delta_sparse(data,RBM,weights=None,q=2): # Pour Bernoulli
	if not RBM.hidden in ['Bernoulli','Spin']:
		print 'delta not supported'
		return None
	else:
	    h = RBM.mean_hiddens(data)
	    if RBM.hidden == 'Spin':
	        h = (1.+h)/2
	    mu = utilities.average(data,c=21,weights=weights)
	    mu_h = utilities.average(h,weights=weights)
	    var_h = utilities.average(h**2,weights=weights) - mu_h**2
	    coactivation = utilities.average_product(h,data,c2=21,weights=weights)
	    covariance = coactivation  - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
	    N = data.shape[0]
	    null_covariance = np.sqrt( (N-1.0)/N**2 * (mu * (1-mu))[np.newaxis,:,:] * var_h[:,np.newaxis,np.newaxis] )
	    significant_indices = np.abs(covariance) > q * null_covariance
	    delta = covariance/((mu_h * (1-mu_h) )[:,np.newaxis,np.newaxis])
	    delta *= significant_indices
	    return delta

def get_kl_divergence(data,RBM,weights=None): # Pour Bernoulli
	if not RBM.hidden in ['Bernoulli','Spin']:
		print 'delta not supported'
		return None
	else:
	    epsilon = 1e-2
	    h = RBM.mean_hiddens(data)
	    if RBM.hidden =='Spin':
	        h = (1.+h)/2.
	    mu = utilities.average(data,c=21,weights=weights)
	    mu_h = utilities.average(h,weights=weights)
	    coactivation = utilities.average_product(h,data,c2=21,weights=weights)
	    cross_entropy = mu[np.newaxis,:,:] * (np.log((epsilon+coactivation)/(epsilon + mu[np.newaxis,:,:] - coactivation)) - np.log( (epsilon+mu_h)/(epsilon+1-mu_h))[:,np.newaxis,np.newaxis] )
	    return cross_entropy,mu,mu_h,coactivation



def test_PGM(PGM,data,data_test,logpnat,logpnat_test,M=10,n_betas=10000,repeats_PL=None):
	if repeats_PL == None:
		if PGM.n_layers>1:
			repeats_PL = 5
		else:
			repeats_PL = 1
	f = PGM.free_energy(data)
	f_test = PGM.free_energy(data_test)
	r2 = np.corrcoef(f,logpnat)[0,1]**2
	r2_test = np.corrcoef(f_test,logpnat_test)[0,1]**2
	overfit = ( f_test.mean() - f.mean() )/f.std() * np.sqrt(data_test.shape[0])
	tmp = np.array([PGM.pseudo_likelihood(data).mean() for _ in range(repeats_PL)])
	tmp_test = np.array([PGM.pseudo_likelihood(data_test).mean() for _ in range(repeats_PL)])
	pl = tmp.mean()
	pl_test = tmp_test.mean()
	pl_std = tmp.std()
	pl_test_std = tmp_test.std()
	PGM.AIS(M=M,n_betas=n_betas)
	l = (-f.mean() - PGM.log_Z_AIS)/data.shape[1]
	std_l = (f.std()/np.sqrt(data.shape[0]) + PGM.log_Z_AIS_std)/data.shape[1]
	l_test = (-f_test.mean() - PGM.log_Z_AIS)/data.shape[1]
	std_l_test = (f_test.std()/np.sqrt(data_test.shape[0]) + PGM.log_Z_AIS_std)/data.shape[1]
	print 'r2 = %.3f, r2_test = %.3f, overfit = %.2f, PL = %.4f +- %.4f, PL_test = %.4f +- %.4f '%(r2,r2_test,overfit,pl,pl_std,pl_test,pl_test_std)
	print 'L = %.4f +- %.4f, L_test = %.4f +- %.4f'%(l,std_l,l_test,std_l_test)
	return [r2,r2_test,overfit,pl,pl_test,pl_std,pl_test_std, l, std_l, l_test,std_l_test]






def Sequence_logo(X,data_type=None,figsize=(10,3),ylabel=None,epsilon=1e-4,with_gaps=True,show=True):
    if data_type is None:
        if X.min()>=0:
            data_type='mean'
        else:
            data_type = 'weights'
    if with_gaps:
        return sequence_logo_gaps.Sequence_logo(X,data_type=data_type,figsize=figsize,ylabel=ylabel,epsilon=epsilon,show=show)
    else:
        return sequence_logo.Sequence_logo(X,data_type=data_type,figsize=figsize,ylabel=ylabel,epsilon=epsilon,show=show)


def Sequence_logo_multiple(matrix, data_type=None,figsize=(10,3),ylabel = None,epsilon=1e-4,ncols=1,with_gaps=True,show=True,count_from=0,title=None):
    if data_type is None:
        if matrix.min()>=0:
            data_type='mean'
        else:
            data_type = 'weights'
    if with_gaps:
        return sequence_logo_gaps.Sequence_logo_multiple(matrix,data_type=data_type,figsize=figsize,ylabel=ylabel,epsilon=epsilon,ncols=ncols,show=show,count_from=count_from,title=title)
    else:
        return sequence_logo.Sequence_logo_multiple(matrix,data_type=data_type,figsize=figsize,ylabel=ylabel,epsilon=epsilon,ncols=ncols,show=show,count_from=count_from,title=title)


def Sequence_logo_all(matrix, name='all_Sequence_logo.pdf', nrows = 5, ncols = 2,data_type=None,figsize=(10,3),ylabel = None,title=None,epsilon=1e-4,with_gaps=True):
    if data_type is None:
        if matrix.min()>=0:
            data_type='mean'
        else:
            data_type = 'weights'
    n_plots = matrix.shape[0]
    plots_per_page = nrows * ncols
    n_pages = int(np.ceil(n_plots/float(plots_per_page)))
    rng = np.random.randn(1)[0] # avoid file conflicts in case of multiple threads.
    mini_name = name[:-4]
    for i in range(n_pages):
        if type(ylabel) == list:
            ylabel_ = ylabel[i*plots_per_page:min(plots_per_page*(i+1),n_plots)]
        else:
            ylabel_ = ylabel
        if type(title) == list:
            title_ = title[i*plots_per_page:min(plots_per_page*(i+1),n_plots)]
        else:
            title_ = title
        fig = Sequence_logo_multiple(matrix[plots_per_page*i:min(plots_per_page*(i+1),n_plots)], data_type=data_type,figsize=figsize,ylabel =ylabel_,title=title_,epsilon=epsilon,ncols=ncols,with_gaps=with_gaps,show=False,count_from=plots_per_page*i)
        fig.savefig(mini_name+'tmp_%s_#%s.png'%(rng,i))
        fig.clear()
    command = 'pdfjoin ' + mini_name+'tmp_%s_#*.png -o %s'%(rng,name)
    os.system(command)
    command = 'rm '+mini_name+'tmp_%s_#*.png'%rng
    os.system(command)
    return 'done'








def display_hidden_inputs(hidden_input,hidden_input_test,weights = None,weights_test = None,order=None,ncols=10):
    n_h = hidden_input.shape[1]
    if order is None:
        order = np.arange(n_h)
    if weights is None:
    	weights = np.ones(hidden_input.shape[0])
	if weights_test is None:
		weights_test = np.ones(hidden_input_test.shape[0])

    nrows = int(np.ceil(n_h/float(ncols)))
    fig, ax = plt.subplots(nrows,ncols)
    fig.set_figheight(3 * nrows)
    fig.set_figwidth(30)
    for i in range(n_h):
        x = i/ncols
        y = i%ncols
        if n_h <= ncols:
            ax[y].hist([hidden_input[:,order[i]],hidden_input_test[:,order[i]]],normed=True,weights=[weights,weights_test],bins=100);
        else:
            ax[x,y].hist([hidden_input[:,order[i]],hidden_input_test[:,order[i]]],normed=True,weights=[weights,weights_test],bins=100);
    return fig


def show_hidden_inputs(RBM,data,data_test,weights = None,weights_test = None,subset=None,ncols=10,show_transform=False,show_mean=False,bins=100,show_count=True):
    hidden_input = RBM.vlayer.compute_output(data,RBM.weights)
    hidden_input_test = RBM.vlayer.compute_output(data_test,RBM.weights)
    n_h = RBM.n_h
    if subset is None:
        subset = np.arange(n_h)
        N_plots = n_h
    else:
        N_plots = len(subset)

    if weights is None:
        weights = np.ones(hidden_input.shape[0])
    if weights_test is None:
        weights_test = np.ones(hidden_input_test.shape[0])



    nrows = int(np.ceil(N_plots/float(ncols)))
    fig, ax = plt.subplots(nrows,ncols)
    fig.set_figheight(3 * nrows)
    fig.set_figwidth(3 * ncols)
    if show_transform:
        psi_min = hidden_input.min()
        psi_max = hidden_input.max()
        psi_range = (psi_max-psi_min) * np.arange(0,1+0.01,0.01) + psi_min
        transformed = RBM.hlayer.transform(np.repeat(psi_range[:,np.newaxis],n_h,axis=1))
    if show_mean:
        psi_min = hidden_input.min()
        psi_max = hidden_input.max()
        psi_range = (psi_max-psi_min) * np.arange(0,1+0.01,0.01) + psi_min
        mean = RBM.hlayer.mean_from_inputs(np.repeat(psi_range[:,np.newaxis],n_h,axis=1))

    count = 0
    for i in subset:
        x = count/ncols
        y = count%ncols
        if ((ncols == 1) & (nrows ==1)):
            ax_ = ax
        elif ((ncols >1) & (nrows ==1)):
            ax_ = ax[y]
        elif ((ncols==1) & (nrows >1)):
            ax_ = ax[x]
        else:
            ax_ = ax[x,y]

        ax_.hist([hidden_input[:,i],hidden_input_test[:,i]],normed=True,weights=[weights,weights_test],bins=bins);
        if show_transform:
            ax2 = ax_.twinx()
            ax2.plot(psi_range,transformed[:,i],c='black',linewidth=2)
        if show_mean:
            ax2 = ax_.twinx()
            ax2.plot(psi_range,mean[:,i],c='black',linewidth=2)
        xmin = min(hidden_input[:,i].min(),hidden_input_test[:,i].min() )
        xmax = max(hidden_input[:,i].max(),hidden_input_test[:,i].max() )
        ax_.set_xlim([xmin,xmax])
        if show_count:
            ax_.set_xlabel('Inputs # %s'%(count+1))
        count +=1
    fig.tight_layout()
    return fig



from sklearn.decomposition import PCA

class ProteinPCA(PCA):
    def __init__(self,n_components =2, n_c=21):
        self.n_components = n_components
        self.n_c = n_c
        super(ProteinPCA, self).__init__(n_components=n_components)
    def fit_transform(self,data):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        out = super(ProteinPCA,self).fit_transform(binary_data)
        self.weights = self.components_.reshape([self.n_components,data.shape[1],self.n_c])
        return out
    def fit(self,data):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        super(ProteinPCA,self).fit(binary_data)
        self.weights = self.components_.reshape([self.n_components,data.shape[1],self.n_c])
    def transform(self,data):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        return super(ProteinPCA,self).transform(binary_data)

from sklearn.decomposition import FastICA

class ProteinICA(FastICA):
    def __init__(self,n_components =2, n_c=21,max_iter=1000):
        self.n_components = n_components
        self.n_c = n_c
        super(ProteinICA, self).__init__(n_components=n_components,max_iter=max_iter)
    def fit_transform(self,data,max_iter=1000):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        out = super(ProteinICA,self).fit_transform(binary_data)
        self.weights = self.components_.reshape([self.n_components,data.shape[1],self.n_c])
        return out
    def fit(self,data):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        super(ProteinICA,self).fit(binary_data)
        self.weights = self.components_.reshape([self.n_components,data.shape[1],self.n_c])
    def transform(self,data):
        binary_data = np.zeros([data.shape[0],data.shape[1],self.n_c])
        for c in range(self.n_c):
            binary_data[:,:,c] = (data == c)
        binary_data = binary_data.reshape([data.shape[0],data.shape[1]*self.n_c])
        return super(ProteinICA,self).transform(binary_data)




def aa_color(letter):
    if letter in ['C']:
        return 'green'
    elif letter in ['F','W','Y']:
        #return [128/256., 89/256., 0.,1.]#'brown'
        return [199/256., 182/256., 0.,1.]#'gold'
    elif letter in ['Q','N','S','T']:
        return 'purple'
    elif letter in ['V','L','I','M']:
        return 'black'
    elif letter in ['K','R','H']:
        return 'blue'
    elif letter in  ['D','E']:
        return 'red'
    elif letter in ['A','P','G']:
        return 'grey'
    else:
        return 'yellow'



def assess_convergence(RBM,data,weights=None,N_PT=10,Nsteps=2,Nthermalize=500,Nchains=100,Lchains=100,update_betas=True,with_reg=False,use_fantasy=False,show_r2=True):
    if use_fantasy:
        if RBM.fantasy_v.ndim ==3:
            data_gen = RBM.fantasy_v[0,:,:]
        else:
            data_gen = RBM.fantasy_v[:,:]

    else:
        data_gen,_ = RBM.gen_data(Nthermalize=Nthermalize,Nchains=Nchains,Lchains=Lchains,N_PT=N_PT,update_betas=update_betas)
    h_data = RBM.mean_hiddens(data)
    h_gen = RBM.mean_hiddens(data_gen)
    mu = utilities.average(data,c=RBM.n_cv)
    mu_gen = utilities.average(data_gen,c=RBM.n_cv)

    mu_h = h_data.mean(0)
    mu_h_gen = h_gen.mean(0)

    cov_d = utilities.average_product(h_data,data,c2=RBM.n_cv) - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
    cov_gen = utilities.average_product(h_gen,data_gen,c2=RBM.n_cv) - mu_gen[np.newaxis,:,:] * mu_h_gen[:,np.newaxis,np.newaxis]
    if with_reg:
        l2 = RBM.l2
        l1 = RBM.l1
        l1b = RBM.l1b
        l1c= RBM.l1c
        l1_custom = RBM.l1_custom
        l1b_custom = RBM.l1b_custom
        n_c2 = RBM.n_cv
        W = RBM.weights
        if l2>0:
            cov_gen += l2 * W
        if l1>0:
            cov_gen += l1 * np.sign( W)
        if l1b>0: # NOT SUPPORTED FOR POTTS
            if n_c2 > 1: # Potts RBM.
                cov_gen += l1b * np.sign(W) *  np.abs(W).mean(-1).mean(-1)[:,np.newaxis,np.newaxis]
            else:
                cov_gen += l1b * np.sign( W) * (np.abs(W).sum(1))[:,np.newaxis]
        if l1c>0: # NOT SUPPORTED FOR POTTS
            cov_gen += l1c * np.sign( W) * ((np.abs(W).sum(1))**2)[:,np.newaxis]
        if l1_custom is not None:
            cov_gen += l1_custom * np.sign(W)
        if l1b_custom is not None:
            cov_gen += l1b_custom[0] * (l1b_custom[1]* np.abs(W)).mean(-1).mean(-1)[:,np.newaxis,np.newaxis] *np.sign(W)



    if RBM.n_cv == 21:
        list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y','X']
    else:
        list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y']
    colors_template = np.array([matplotlib.colors.to_rgba(aa_color(letter)) for letter in list_aa] )
    color = np.repeat(colors_template[np.newaxis,:,:],data.shape[1],axis=0).reshape([data.shape[1]* RBM.n_cv,4])

    fig1, ax = plt.subplots()
    plt.scatter(mu.flatten(),mu_gen.flatten(),c=color);
    plt.plot([mu.min(),mu.max()],[mu.min(),mu.max()])
    plt.xlabel('$<v_i^a>$ data');
    plt.ylabel('<v_i^a> model');
    if show_r2:
        r2 = np.corrcoef(mu.flatten(),mu_gen.flatten() )[0,1]**2    
        plt.title('r2 = %.3f'%r2)
    plt.show()

        
    fig2, ax = plt.subplots()    
    plt.scatter(mu_h,mu_h_gen);
    plt.plot([mu_h.min(),mu_h.max()],[mu_h.min(),mu_h.max()])
    plt.xlabel('$<h^\mu>$ data');
    plt.ylabel('<h^\mu> model');
    if show_r2:
        r2 = np.corrcoef(mu_h,mu_h_gen)[0,1]**2
        plt.title('r2 = %.3f'%r2)    
    plt.show()

    fig3, ax = plt.subplots()
    color = np.repeat(np.repeat(colors_template[np.newaxis,np.newaxis,:,:],RBM.n_h,axis=0),data.shape[1] ,axis=1).reshape([data.shape[1] * RBM.n_h * RBM.n_cv,4])
    plt.scatter(cov_d.flatten(),cov_gen.flatten(),c = color);
    plt.plot([cov_d.min(),cov_d.max()],[cov_d.min(),cov_d.max()])
    plt.xlabel('$<v_i^a h^\mu>$ data');
    plt.ylabel('<v_i^a h^\mu> model');
    if show_r2:
        r2 = np.corrcoef(cov_d.flatten(),cov_gen.flatten())[0,1]**2
        plt.title('r2 = %.3f'%r2)    
    plt.show()
    return fig1,fig2,fig3


def assess_convergence_BM(BM,data,weights=None,N_PT=10,Nsteps=2,Nthermalize=500,Nchains=100,Lchains=100,update_betas=True,with_reg=False,use_fantasy=False,show=True,show_r2=True):
    if use_fantasy:
        if BM.fantasy_x.ndim ==3:
            data_gen = BM.fantasy_x[0,:,:]
        else:
            data_gen = BM.fantasy_x[:,:]

    else:
        data_gen = BM.gen_data(Nthermalize=Nthermalize,Nchains=Nchains,Lchains=Lchains,N_PT=N_PT,update_betas=update_betas)

    mu = utilities.average(data,c=BM.n_c)
    mu_gen = utilities.average(data_gen,c=BM.n_c)

    cov_d = utilities.average_product(data,data,c1=BM.n_c,c2=BM.n_c) - mu[np.newaxis,:,np.newaxis,:] * mu[:,np.newaxis,:,np.newaxis]
    cov_gen = utilities.average_product(data_gen,data_gen,c1=BM.n_c,c2=BM.n_c) - mu_gen[np.newaxis,:,np.newaxis,:] * mu_gen[:,np.newaxis,:,np.newaxis]
    if with_reg:
        l2 = BM.l2
        l1 = BM.l1
        J = BM.layer.fields
        if l2>0:
            cov_gen += l2 * J
        if l1>0:
            cov_gen += l1 * np.sign( W)


    if BM.n_c == 21:
        list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y','X']
    else:
        list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y']
    try:
        colors_template = np.array([matplotlib.colors.to_rgba(aa_color(letter)) for letter in list_aa] )
        color = np.repeat(colors_template[np.newaxis,:,:],BM.N,axis=0).reshape([BM.N* BM.n_c,4])        
    except:
        color = None



    fig1,ax = plt.subplots()
    plt.scatter(mu.flatten(),mu_gen.flatten(),c=color);
    plt.plot([mu.min(),mu.max()],[mu.min(),mu.max()])
    plt.xlabel('$<v_i^a>$ data');
    plt.ylabel('$<v_i^a>$ model');
    if show_r2:
        r2 = np.corrcoef(mu.flatten(),mu_gen.flatten() )[0,1]**2    
        plt.title('r2 = %.3f'%r2)    
    if show:
        plt.show()
    try:
        color = np.repeat(np.repeat(np.repeat(colors_template[np.newaxis,np.newaxis,np.newaxis,:,:],BM.N,axis=0),BM.N ,axis=1),BM.n_c,axis=2).reshape([BM.N**2*BM.n_c**2,4])
    except:
        color = None
    fig2,ax = plt.subplots()    
    plt.scatter(cov_d.flatten(),cov_gen.flatten(),c = color);
    plt.plot([cov_d.min(),cov_d.max()],[cov_d.min(),cov_d.max()])
    plt.xlabel('$<v_i^a v_j^b>$ data');
    plt.ylabel('$<v_i^a v_j^b>$ model');
    if show_r2:
        r2 = np.corrcoef(cov_d.flatten(),cov_gen.flatten())[0,1]**2
        plt.title('r2 = %.3f'%r2)    
    if show:
        plt.show()
    return fig1,fig2

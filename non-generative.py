__author__ = "Galin Georgiev"
__copyright__ = "Copyright 2015, Gamma Dynamics, LLC"
__version__ = "1.0.0.0"


import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import scipy.io
import time
from toolbox import *
from optimizer import adam 
import theano.tensor.slinalg

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.sort import argsort, ArgSortOp

#hyper-parameters and data
binarized = False #external binarization
num_epochs = 10000 
batch_size = 10000 #batch size

num_train = 60000 # num training observations
init_lr = 0.0015 #initial learning rate
it_lr = init_lr #learning rate
lr_halflife = 500 #half-life of learning rate
gaussian_err = False

#import data
P, trX, teX, trY, teY = mnist(binarized, distort=False, shuffle=False, shuffle_test_labels = False, ntrain=num_train,ntest=10000, onehot=True)   

#layer sizes
n_x = trX.shape[1] #number of observables
n_h = 700 
n_h2 = 700
n_h3 = 700
n_h4 = 700

#model
def model_NG_ACE(X, n_b, gaussian_err, p_drop_input,  p_drop_hidden):
		
	X = dropout(X, p_drop_input)
	h = dropout((T.tensordot(X, W_h, [[1],[1]]) + b_h).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h
	h = batchnorm(h, epsilon= 0.) 
			
	#dual reconstruction err
	if gaussian_err:
		dual_X_hat = T.dot(h,T.dot(h.T,X)) /n_b 
		dual_recon_err = 0.5 * T.sqr(X - dual_X_hat).sum() 
	else:
		dual_X_hat =  T.nnet.sigmoid(T.dot(h,T.dot(h.T,X))/n_b) #shape: batch_size x n_x
		dual_recon_err = T.nnet.binary_crossentropy(dual_X_hat,X).sum() 
		
	h2 = dropout((T.tensordot(h, W_h2, [[1],[1]]) + b_h2).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h2 
	h3 = dropout ((T.tensordot(h2, W_h3, [[1],[1]]) + b_h3).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h3
	h3 = batchnorm(h3, epsilon=0)
	h4 = dropout ((T.tensordot(h3, W_h4, [[1],[1]]) + b_h4).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h4
					
	prob_y = softmax(T.dot(h4, W_o)) #classification probabilities
	return [prob_y , dual_recon_err]
		
#symbolic variables	
X = T.fmatrix()
Y = T.fmatrix()
learning_rate = T.fscalar('learning_rate')
start, end = T.lscalars('start', 'end')

#weights and biases initialization
W_h = shared_normal((2, n_x, n_h))  
W_h2 = shared_normal((2, n_h, n_h2))
W_h3 = shared_normal((2, n_h2, n_h3)) 
W_h4 = shared_normal((2, n_h3, n_h4))
W_o  = shared_normal((n_h4, 10))
b_h = shared_normal((2, n_h), sigma = 0)  
b_h2 = shared_normal((2, n_h2), sigma = 0)
b_h3 = shared_normal((2, n_h3), sigma = 0)
b_h4 = shared_normal((2, n_h4), sigma = 0)

X = binomial(X) #internal binarization
#model calls	
[dout_prob_y, dout_dual_recon_err] = model_NG_ACE(X, batch_size, gaussian_err, 0.2, 0.5)  #with dropout
[prob_y, dual_recon_err] = model_NG_ACE(X, batch_size, gaussian_err, 0., 0.) #without dropout
	
y_model = T.argmax(prob_y, axis=1) #model labels
#dropout classification err
dout_class_err =  T.nnet.categorical_crossentropy(dout_prob_y, Y).sum() 

#optimizer	call
cost = dout_class_err + dout_dual_recon_err 
params = [W_h, W_h2, W_h3, W_h4, W_o, b_h, b_h2, b_h3, b_h4] 
updates, norm_grad = adam(cost, params, lr = learning_rate, data_part = float(batch_size) / P)

#givens
s_trX, s_teX, s_trY, s_teY = shared(trX), shared(teX), shared(trY), shared(teY)   
tr_batch_X = s_trX[start : end]
tr_batch_Y = s_trY[start : end]
te_batch_X = s_teX[start : end]
te_batch_Y = s_teY[start : end] 
	
#train & test functions
mode = theano.compile.get_default_mode()	
train = theano.function(inputs=[start, end, learning_rate],  outputs= [dout_class_err, dual_recon_err,   y_model, norm_grad], updates=updates,  givens = {X : tr_batch_X, Y : tr_batch_Y}, allow_input_downcast=True, mode = mode) 
test = theano.function(inputs=[start, end], outputs=  [dual_recon_err,    y_model], givens = {X : te_batch_X}, allow_input_downcast=True,  mode = mode) 

#main loop over epochs	
tr_len = len(trY)
te_len = len(teY) 
for it in range(num_epochs):
	begin = time.time()

	#one epoch of training set
	tr_epoch_class_err = 0.
	tr_epoch_dual_recon_err = 0.
	tr_epoch_y_model = 0.
	tr_epoch_norm_grad = 0.
	tr_batches = np.arange(0,tr_len, batch_size)
	if tr_batches[-1] != tr_len:
		tr_batches = np.append(tr_batches,tr_len)
   	for i in xrange(0,len(tr_batches) - 1):
		[tr_batch_class_cost, tr_batch_dual_recon_err,  tr_batch_y_model, tr_batch_norm_grad] = train( tr_batches[i], tr_batches[i + 1], it_lr) 
		tr_epoch_class_err += tr_batch_class_cost
		tr_epoch_dual_recon_err += tr_batch_dual_recon_err
		tr_epoch_y_model +=  np.sum(np.argmax(trY[tr_batches[i]:tr_batches[i + 1]], axis=1) == tr_batch_y_model)
		tr_epoch_norm_grad += tr_batch_norm_grad
	it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
	
	#one epoch of test set
	te_epoch_dual_recon_err = 0.
	te_epoch_y_model = 0.
	te_batches = np.arange(0, te_len, batch_size)
	if te_batches[-1] != te_len:
		te_batches = np.append(te_batches,te_len)
	for i in xrange(0,len(te_batches) - 1): 
		[te_batch_dual_recon_err,   te_batch_y_model] = test( te_batches[i], te_batches[i + 1])
		te_epoch_dual_recon_err += te_batch_dual_recon_err
		te_epoch_y_model +=  np.sum(np.argmax(teY[te_batches[i]:te_batches[i + 1]], axis=1) == te_batch_y_model)
		
		
	end = time.time()
				
	print("%d,%.4f,%.2f,%.2f,%.4f,%.4f,%.2e, %.2f" % (it, tr_epoch_class_err/tr_len, tr_epoch_dual_recon_err/tr_len, te_epoch_dual_recon_err/ te_len, 1 - tr_epoch_y_model / tr_len, 1 - te_epoch_y_model / te_len, tr_epoch_norm_grad/tr_len, end - begin))
		
		


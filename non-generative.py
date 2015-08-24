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
import theano.tensor.slinalg

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.sort import argsort, ArgSortOp

binarized = False
num_epochs = 100000 
tr_batch_size = 10000 #training batch size
te_batch_size = 10000 #testing batch size

num_train = 60000 # num training observations
init_lr = 0.0015 #initial learning rate
it_lr = init_lr #learning rate
lr_halflife = 500 #half-life of learning rate




srnd = MRG_RandomStreams()

#Optimizer
#-------------------------------------------------------------------------------------------------- 
def adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, decay_rate=0.0001, data_part=0.0): 
    updates = []
    grads = T.grad(cost, params)
    i = shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1) ** i_t
    fix2 = 1. - (1. - b2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = shared(p.get_value() * 0.)
        v = shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t) - p * decay_rate * data_part
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates, norm_gs(params, grads)



# Data
#--------------------------------------------------------------------------------------------------
P, trX, teX, trY, teY = mnist(binarized, distort=False, shuffle=False, shuffle_test_labels = False, ntrain=num_train,ntest=10000, onehot=True)   

#layer sizes
n_x = trX.shape[1] #number of observables
n_h = 700 
n_h2 = 700
n_h3 = 700
n_h4 = 700
n_c = 2 #use for non-linearities
gaussian = False

#model
def model_NG_ACE(X, n_p, gaussian, p_drop_input,  p_drop_hidden):
		
	X = dropout(X, p_drop_input)
	h = dropout((T.tensordot(X, w_h, [[1],[1]]) + b_h).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x n_c x n_h
	h = batchnorm(h, epsilon= 0.) 

		
	#dual auto-encoder error
	if gaussian:
		phx = T.dot(h,T.dot(h.T,X)) /n_p 
		log_phx = 0.5 * T.sqr(X - phx).sum(axis=0).sum() 
	else:
		phx =  T.nnet.sigmoid(T.dot(h,T.dot(h.T,X))/n_p) #shape: batch_size x n_x
		log_phx = T.nnet.binary_crossentropy(phx,X).sum(axis=0).sum() 
		
	h2 = dropout((T.tensordot(h, w_h2, [[1],[1]]) + b_h2).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x n_c x n_h2 
	h3 = dropout ((T.tensordot(h2, w_h3, [[1],[1]]) + b_h3).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x n_c x n_h3
	h3 = batchnorm(h3, epsilon=0)
	h4 = dropout ((T.tensordot(h3, w_h4, [[1],[1]]) + b_h4).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x n_c x n_h4
					
	py_x = softmax(T.dot(h4, w_o))
	return [py_x , log_phx]
		
#symbolic variables	
X = T.fmatrix()
Y = T.fmatrix()
learning_rate = T.fscalar('learning_rate')
start, end = T.lscalars('start', 'end')

#weights and biases
w_h = shared_normal((n_c, n_x, n_h))  #shared_normal((n_x, n_h))  
w_h2 = shared_normal((n_c, n_h, n_h2))
w_h3 = shared_normal((n_c, n_h2, n_h3)) 
w_h4 = shared_normal((n_c, n_h3, n_h4))
w_o  = shared_normal((n_h4, 10))
b_h = shared_normal((n_c, n_h), sigma = 0) #b_h = shared_normal((n_h,), sigma = 0) 
b_h2 = shared_normal((n_c, n_h2), sigma = 0)
b_h3 = shared_normal((n_c, n_h3), sigma = 0)
b_h4 = shared_normal((n_c, n_h4), sigma = 0)

#binarization
X = binomial(X)
	
[dout_py_x, dout_log_phx] = model_NG_ACE(X, tr_batch_size, gaussian, 0.2, 0.5)  #with dropout
[py_x, log_phx] = model_NG_ACE(X, tr_batch_size, gaussian, 0., 0.) #without dropout
	
y_x = T.argmax(py_x, axis=1) #model labels
if gaussian:
	dout_class_cost = 0.5 * T.sqr(dout_py_x - Y).sum(axis=0).sum() 
else:
	dout_class_cost = (-Y * T.log(dout_py_x)).sum(axis=0).sum() # equivalent to T.nnet.categorical_crossentropy(noise_py_x, Y).sum() 
	
cost = dout_class_cost + dout_log_phx 
params = [w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4] 
updates, norm_grad = adam(cost, params, lr = learning_rate, data_part = float(tr_batch_size) / P)
mode = theano.compile.get_default_mode()

#givens
s_trX, s_teX, s_trY, s_teY = shared(trX), shared(teX), shared(trY), shared(teY)   
tr_batch_X = s_trX[start : end]
tr_batch_Y = s_trY[start : end]
te_batch_X = s_teX[start : end]
te_batch_Y = s_teY[start : end] 
	
#train & test functions	
train = theano.function(inputs=[start, end, learning_rate],  outputs= [dout_class_cost, log_phx,   y_x, norm_grad], updates=updates,  givens = {X : tr_batch_X, Y : tr_batch_Y}, on_unused_input = 'ignore', allow_input_downcast=True, mode = mode) 
test = theano.function(inputs=[start, end], outputs=  [log_phx,    y_x], givens = {X : te_batch_X, Y : te_batch_Y}, on_unused_input = 'ignore',allow_input_downcast=True,  mode = mode) 

#main epoch loop	
tr_len = len(trY)  
for it in range(num_epochs):
	begin = time.time()

	#one epoch on training set
	tr_class_cost_cum = 0.
	tr_log_phx_cum = 0.
	tr_class_cum = 0.
	tr_norm_grad_cum = 0.
	tr_batches = np.arange(0,tr_len, tr_batch_size)
	if tr_batches[-1] != tr_len:
		tr_batches = np.append(tr_batches,tr_len)
   	for i in xrange(0,len(tr_batches) - 1):
		[tr_batch_class_cost, tr_batch_log_phx,  tr_batch_class, tr_batch_norm_grad] = train( tr_batches[i], tr_batches[i + 1], it_lr) 
		tr_class_cost_cum += tr_batch_class_cost
		tr_log_phx_cum += tr_batch_log_phx
		tr_class_cum +=  np.sum(np.argmax(trY[tr_batches[i]:tr_batches[i + 1]], axis=1) == tr_batch_class)
		tr_norm_grad_cum += tr_batch_norm_grad
	it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
	
	#one epoch on test set
	te_len = len(teY)
	te_log_phx_cum = 0.
	te_class_cum = 0.
	te_batches = np.arange(0, te_len, te_batch_size)
	if te_batches[-1] != te_len:
		te_batches = np.append(te_batches,te_len)
	for i in xrange(0,len(te_batches) - 1): 
		[te_batch_log_phx,   te_batch_class] = test( te_batches[i], te_batches[i + 1])
		te_log_phx_cum += te_batch_log_phx
		te_class_cum +=  np.sum(np.argmax(teY[te_batches[i]:te_batches[i + 1]], axis=1) == te_batch_class)
		
		
	end = time.time()
				
	print("%d,%.4f,%.2f,%.2f,%.4f,%.4f,%.2e, %.2f" % (it, tr_class_cost_cum/tr_len, tr_log_phx_cum/tr_len, te_log_phx_cum/ len(teY), 1 - tr_class_cum / tr_len, 1 - te_class_cum / te_len, tr_norm_grad_cum/tr_len, end - begin))
		
		


import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.ifelse import ifelse
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import max_pool_2d_same_size
import numpy as np
import scipy.io
import time
from toolbox import *
from optimizer import adam
import theano.tensor.slinalg


from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.sort import argsort, ArgSortOp

task = 2 #1 density estimation, 2 classification
sampling_class = 2  #1  Gaussian, 2  Laplacian 

#hyper-parameters
if task == 1:
	batch_size = 1000 #batch size
	binarized = True #external binarization
	init_lr = 0.0002 #initial learning rate
elif task == 2:
	batch_size = 10000
	binarized = False
	init_lr = 0.0015
	
num_epochs = 10000
num_train = 60000 # num training observations
it_lr = init_lr #learning rate
lr_halflife = 500 #half-life of learning rate
gaussian_err = False


#import data
P, trX, teX, trY, teY = mnist(binarized, distort=False, shuffle=False, shuffle_test_labels = False, ntrain=num_train,ntest=10000, onehot=True)   

#layer sizes
n_x = trX.shape[1]
n_h = 700 
n_h2 = 700
n_h3 = 700
n_h4 = 700
if task == 1:
	n_z = 400
elif task == 2:
	n_z = 100

n_y = trY.shape[1]
n_c = n_y

#model	
def model_G_ACE(X, X_stack, Y, n_b, gaussian_err, p_drop_input,  p_drop_hidden):
		

	X = dropout(X, p_drop_input)
		
	h0 = dropout(T.tanh(T.dot(X, W_h) + b_h), p_drop_hidden) #shape: batch_size x n_h
	h = batchnorm(h0, epsilon= 0.) 
	
	#classifier branch	
	#dual treconstruction error
	if gaussian_err:
		dual_X_hat = T.dot(h,T.dot(h.T,X)) /n_b 
		dual_recon_err = 0.5 * T.sqr(X - dual_X_hat).sum(axis=0).sum() 
	else:
		dual_X_hat =  T.nnet.sigmoid(T.dot(h,T.dot(h.T,X))/n_b) #shape: batch_size x n_x
		dual_recon_err = T.nnet.binary_crossentropy(dual_X_hat,X).sum() 
				
	h2 = dropout((T.tensordot(h, W_h2, [[1],[1]]) + b_h2).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h2 
	h3 = dropout ((T.tensordot(h2, W_h3, [[1],[1]]) + b_h3).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h3
	h3 = batchnorm(h3, epsilon=0)
	h4 = dropout ((T.tensordot(h3, W_h4, [[1],[1]]) + b_h4).max(axis = 1), p_drop_hidden) #shape inside tensordot: batch_size x 2 x n_h4
	prob_y = softmax(T.dot(h4, W_o)) #classification probabilities; shape: batch_size x n_c
		

	#auto-encoder branch 
	#gen latent layer
	mu = T.tensordot(h0, W_mu, [[1],[1]])  + b_mu #shape: batch_size x n_c x n_z
	log_sigma = 0.5 * (T.tensordot(h0, W_sigma, [[1],[1]]) + b_sigma)
		
	#sampling
	srnd = MRG_RandomStreams()
	if sampling_class == 1: #Gaussian
		eps = srnd.normal(mu.shape, dtype=theano.config.floatX)
		z = mu + eps * T.exp(log_sigma) #shape: batch_size x n_c x n_z
	elif sampling_class == 2: #Laplacian
		eps0 = srnd.uniform(mu.shape, dtype=theano.config.floatX)
		if T.lt(eps0 , 0.5) == 1:
			z = mu + T.exp(log_sigma) * T.sqrt(0.5) * T.log(eps0 + eps0)
		else:
			z = mu - T.exp(log_sigma) * T.sqrt(0.5) * T.log(2.0 - eps0 - eps0) #shape: batch_size x n_c x n_z
			
	#gen error
	if sampling_class == 1: #Gaussian
		gen_err_stack = -0.5*  (1 + 2*log_sigma - mu**2 -T.exp(2*log_sigma)).sum(axis=2) #shape: batch_size x n_c
	elif sampling_class == 2: #Laplacian
		gen_err_stack = (  - log_sigma  + T.abs_(mu) /T.sqrt(0.5) + T.exp(log_sigma)*T.exp(- T.abs_(mu)/T.exp(log_sigma)/T.sqrt(0.5)) - 1 ).sum(axis=2)  
	gen_err = (gen_err_stack * prob_y).sum()  #Unsupervised:upper bound for mixture of densities, using weights prob_y
	sup_gen_err = (gen_err_stack * Y).sum() #Supervised
		

	## decoder
	h_dec = dropout(T.tanh(T.batched_dot(z.dimshuffle((1,0,2)), W_h_dec).dimshuffle((1,0,2))   + b_h_dec),  p_drop_hidden)  
	h2_dec = T.batched_dot(h_dec.dimshuffle((1,0,2)), W_h2_dec).dimshuffle((1,0,2)) + b_h2_dec  
		
	if gaussian_err:
		X_hat = rectify(T.batched_dot(h_dec.dimshuffle((1,0,2)), W5).dimshuffle((1,0,2)) + b5)
		recon_err_stack = 0.5 * np.log(2 * np.pi) +  0.5 * T.sqr(X_stack - X_hat).sum(axis = 2)  
		recon_err = (recon_err_stack * prob_y).sum() 
		
	else:
		X_hat = T.nnet.sigmoid(h2_dec) #shape: batch_size x n_c x n_x
		recon_err_stack = T.nnet.binary_crossentropy(X_hat,X_stack).sum(axis = 2)  #shape: batch_size x n_c
		recon_err = (recon_err_stack * prob_y).sum() #Unsupervised: mixture of densities, using  weights prob_y
		sup_recon_err = (recon_err_stack * Y).sum()   #Supervised
			
	return [ dual_recon_err,  gen_err, sup_gen_err, X_hat, recon_err, sup_recon_err, prob_y]
	

#symbolic variables
X = T.fmatrix()
Y = T.fmatrix()
Z = T.ftensor3()
learning_rate = T.fscalar('learning_rate')
start, end = T.lscalars('start', 'end')	

#weights and biases initialization
#1.auto-encoder branch
W_mu = shared_normal((n_c, n_h, n_z))
W_sigma = shared_normal((n_c, n_h, n_z))
W_h_dec = shared_normal((n_c, n_z, n_h)) 
W_h2_dec = shared_normal((n_c, n_h, n_x)) 
b_mu = shared_normal((n_c, n_z), sigma = 0)
b_sigma = shared_normal((n_c, n_z), sigma = 0)
b_h_dec = shared_normal((n_c, n_h), sigma = 0) 
b_h2_dec  = shared_normal((n_c, n_x), sigma = 0) 

#2.classifier branch	
W_h = shared_normal((n_x, n_h)) 
W_h2 = shared_normal((2, n_h, n_h2))
W_h3 = shared_normal((2, n_h2, n_h3)) 
W_h4 = shared_normal((2, n_h3, n_h4))
W_o  = shared_normal((n_h4, n_c))
b_h = shared_normal((n_h,), sigma = 0) 
b_h2 = shared_normal((2, n_h2), sigma = 0)
b_h3 = shared_normal((2, n_h3), sigma = 0)
b_h4 = shared_normal((2, n_h4), sigma = 0)

X = binomial(X) #internal binarization
X_stack = T.stack(X,X,X,X,X,X,X,X,X,X).dimshuffle((1,0,2))
#model calls		
[ dout_dual_recon_err,  dout_gen_err, dout_sup_gen_err, dout_X_hat, dout_recon_err, dout_sup_recon_err, dout_prob_y] = model_G_ACE(X, X_stack, Y, batch_size, gaussian_err, 0.2, 0.5) #with dropout
[ dual_recon_err, gen_err, sup_gen_err, X_hat, recon_err, sup_recon_err, prob_y] = model_G_ACE(X, X_stack, Y, batch_size, gaussian_err, 0., 0.) #without dropout
	
y_model = T.argmax(prob_y, axis=1) #model labels
#classification err
class_err =  T.nnet.categorical_crossentropy(prob_y, Y).sum() #(-Y * T.log(prob_y)).sum(axis=0).sum()
dout_class_err =  T.nnet.categorical_crossentropy(dout_prob_y, Y).sum() #(-Y * T.log(dout_prob_y)).sum(axis=0).sum()

#optimizer call
if task == 1:
	cost = sup_recon_err + sup_gen_err + dout_class_err 
elif task == 2:
	cost = sup_recon_err + sup_gen_err + dout_class_err + dout_dual_recon_err
params = [W_h, W_h2, W_h3, W_h4, W_o, W_mu, W_sigma, W_h_dec, W_h2_dec,  b_h, b_h2, b_h3, b_h4, b_mu, b_sigma, b_h_dec, b_h2_dec] 
updates, norm_grad = adam(cost, params, lr = learning_rate, data_part = float(batch_size) / P)
	
#supervised non-generative output
sup_X_hat = (X_hat*Y.dimshuffle((0,1,'x')) ).sum(axis=1)
	
#supervised generative output
h_dec_gen =   T.tanh(T.batched_dot(Z.dimshuffle((1,0,2)), W_h_dec).dimshuffle((1,0,2)) + b_h_dec) 
if gaussian_err:
	sup_X_hat_gen = (rectify(T.batched_dot(h_dec_gen.dimshuffle((1,0,2)), W_h2_dec).dimshuffle((1,0,2)) + b_h2_dec)  * Y.dimshuffle((0,1,'x')) ).sum(axis=1)
else:
	sup_X_hat_gen = (T.nnet.sigmoid(T.batched_dot(h_dec_gen.dimshuffle((1,0,2)), W_h2_dec).dimshuffle((1,0,2)) + b_h2_dec) * Y.dimshuffle((0,1,'x')) ).sum(axis=1) 

#givens
s_trX, s_teX, s_trY, s_teY = shared(trX), shared(teX), shared(trY), shared(teY)
tr_batch_X = s_trX[start : end]
tr_batch_Y = s_trY[start : end]
te_batch_X = s_teX[start : end]
te_batch_Y = s_teY[start : end]

if sampling_class == 1: #Gaussian
	s_geZ = shared(np.random.randn(P, n_c, n_z))
elif sampling_class == 2: #Laplacian
	s_geZ = shared(np.random.laplace(loc=0.0, scale=np.sqrt(0.5),size= (P, n_c, n_z)) )
batch_geZ = s_geZ[start : end] 

#train ,test & generate functions
mode = theano.compile.get_default_mode()		
train = theano.function(inputs=[start, end, learning_rate], outputs=[recon_err, gen_err, class_err,  dual_recon_err, y_model, norm_grad], updates=updates, givens = {X : tr_batch_X, Y : tr_batch_Y}, allow_input_downcast=True, on_unused_input = 'ignore',mode=mode)
test  = theano.function(inputs=[start, end], outputs=[recon_err, gen_err, class_err,  dual_recon_err, y_model],                  givens = {X : te_batch_X, Y : te_batch_Y}, allow_input_downcast=True, on_unused_input = 'ignore',mode=mode)
reconstruct  = theano.function(inputs=[start, end], outputs=sup_X_hat,                  givens = {X : te_batch_X, Y : te_batch_Y}, allow_input_downcast=True, on_unused_input = 'ignore',mode=mode)
generate = theano.function(inputs=[start, end], outputs=sup_X_hat_gen, givens = {Z : batch_geZ, Y : te_batch_Y}, allow_input_downcast=True, on_unused_input = 'ignore', mode=mode)

rnds = RandomStreams()

handle = 0
		
#main  loop	over epochs
tr_len = len(trY) 
te_len = len(teY) 
for it in range(num_epochs):
	begin = time.time()

	#one epoch of training set
	tr_epoch_recon_err = 0.
	tr_epoch_gen_err = 0.
	tr_epoch_class_err = 0.
	tr_epoch_dual_recon_err = 0.
	tr_epoch_y_model = 0.
	tr_epoch_norm_grad = 0.
	tr_batches = np.arange(0,tr_len,batch_size)
	if tr_batches[-1] != tr_len:
		tr_batches = np.append(tr_batches,tr_len)
	for i in xrange(0,len(tr_batches) - 1): 
		[tr_batch_recon_err, tr_batch_gen_err, tr_batch_class_err, tr_batch_dual_recon_err, tr_batch_y_model, tr_batch_norm_grad] = train( tr_batches[i], tr_batches[i + 1], it_lr)
		tr_epoch_recon_err += tr_batch_recon_err
		tr_epoch_gen_err += tr_batch_gen_err
		tr_epoch_class_err += tr_batch_class_err
		tr_epoch_dual_recon_err += tr_batch_dual_recon_err
		tr_epoch_y_model +=  np.sum(np.argmax(trY[tr_batches[i]:tr_batches[i + 1]], axis=1) == tr_batch_y_model)
		tr_epoch_norm_grad += tr_batch_norm_grad
	
	#one epoch of test set	
	te_epoch_recon_err = 0.
	te_epoch_gen_err = 0.
	te_epoch_class_err = 0.
	te_epoch_dual_recon_err = 0.
	te_epoch_y_model = 0.
	te_batches = np.arange(0, te_len, batch_size)
	if te_batches[-1] != te_len:
		te_batches = np.append(te_batches,te_len)	
	for i in xrange(0,len(te_batches) - 1): 
		[te_batch_recon_err, te_batch_gen_err, te_batch_class_err, te_batch_dual_recon_err, te_batch_y_model] = test( te_batches[i], te_batches[i + 1])
		te_epoch_recon_err += te_batch_recon_err
		te_epoch_gen_err += te_batch_gen_err
		te_epoch_class_err += te_batch_class_err
		te_epoch_dual_recon_err += te_batch_dual_recon_err
		te_epoch_y_model += np.sum(np.argmax(teY[te_batches[i]:te_batches[i + 1]], axis=1) == te_batch_y_model)
			
	it_lr = float(it_lr*np.power(0.5, 1./lr_halflife))
	end = time.time()
	print("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.2e,%.2f" % (it, tr_epoch_recon_err / tr_len,  te_epoch_recon_err / te_len, tr_epoch_gen_err / tr_len, te_epoch_gen_err / te_len,  tr_epoch_class_err/tr_len,  tr_epoch_dual_recon_err/tr_len, te_epoch_dual_recon_err/ te_len, 1 - tr_epoch_y_model / tr_len,  1 - te_epoch_y_model / te_len, tr_epoch_norm_grad/tr_len, end - begin)) 
	
	# Generate samples
	if it % 20 == 0:
		y_samples = generate(0,900)
		y_reconstuct = reconstruct(0, 300)
		y_samples[300:600] = y_reconstuct[0:300]
		y_samples[600:900] = teX[0:300]
		handle = visualize(it, y_samples[0:len(y_samples)], [30,30], 'samples_',handle)




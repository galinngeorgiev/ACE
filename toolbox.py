__author__ = "Galin Georgiev"
__copyright__ = "Copyright 2015, Gamma Dynamics, LLC"
__version__ = "1.0.0.0"


import numpy, os, cPickle
from PIL import Image
import matplotlib.pyplot as plt
from subprocess import Popen
import numpy as np
import os
import gzip
from random import shuffle

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

srnd = MRG_RandomStreams()

datasets_dir = 'C:\\Users\\galin.georgiev\\LightVerge\\Datasets\\' #'C:\\AI\Input\\VisualRecognition\\'



def shared(X, name=None, dtype=theano.config.floatX, borrow=False, broadcastable=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name, borrow=borrow, broadcastable=broadcastable)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def softmax(X):
	if X.ndim ==2:
		e_x = T.exp(X - X.max(axis = -1).dimshuffle(0, 'x'))
		return e_x / e_x.sum(axis = -1).dimshuffle(0, 'x')
	elif X.ndim ==3:
		e_x = T.exp(X - X.max(axis = -1).dimshuffle(0, 1, 'x'))
		return e_x / e_x.sum(axis = -1).dimshuffle(0, 1, 'x')

def rectify(X):
    return T.maximum(X, 0.)

def rectify_min(X):
    return T.minimum(X, 0.)

def binomial(X):
    return srnd.binomial(X.shape, p=X, dtype=theano.config.floatX)

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srnd.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def gaussian(X, std=0.):
    if std > 0:
        X += srnd.normal(X.shape, std = std, dtype=theano.config.floatX)
    return X

def shared_zeros(shape, dtype=theano.config.floatX, name=None, broadcastable=None):
    return shared(np.zeros(shape), dtype=dtype, name=name, broadcastable=broadcastable)

def shared_ones(shape, dtype=theano.config.floatX, name=None, broadcastable=None):
    return shared(np.ones(shape), dtype=dtype, name=name, broadcastable=broadcastable)

def shared_uniform(shape, sv_adjusted=True, scale=1):
	if (sv_adjusted):
		if (len(shape) == 1):
			scale_factor = scale * 0.5 / np.sqrt(shape[0])
		else:
			scale_factor = scale * np.sqrt(6)/ (np.sqrt(shape[0] + shape[1]))
	else:
			scale_factor = scale
	return shared(np.random.uniform(low=-scale, high=scale, size=shape))

def shared_uniform(shape, range=[-0.05,0.05]):
    return shared(np.random.uniform(low=range[0], high=range[1], size=shape))

def shared_normal(shape, sv_adjusted=True, sigma=1.0):
    if (sv_adjusted):
        if (len(shape) == 1):
            sigma_factor = sigma / np.sqrt(shape[0])
        else:
            sigma_factor = sigma / (np.sqrt(shape[0]) + np.sqrt(shape[1]))
    else:
        sigma_factor = sigma    
    return shared(np.random.standard_normal(shape) * sigma_factor) #shared(np.random.randn(*shape) * sigma_factor)

def norm_gs(params, grads):
    norm_gs = 0.
    for g in grads:
        norm_gs += (g**2).sum()
    
    return norm_gs

#batch normalization (basic) 
def batchnorm(h, epsilon=0.):
	
	m = T.mean(h, axis=0, keepdims=True) #h = h - T.mean(h,axis =0)  #de-mean every column
	std = T.sqrt(T.mean(h*h,axis =0) + epsilon) #T.sqrt(T.var(h, axis = 0, keepdims=True) + epsilon) #inv_h_norm = 1/T.mean(h*h,axis =0) #1/T.sum(h*h,axis =0) # 
	h = (h - m) /std #h * T.sqrt(inv_h_norm) #T.dot(h,T.nlinalg.diag(T.sqrt(inv_h_norm))) #normalize every node
	return h


def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def shuffledata(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [numpy.matrix([d[idx] for idx in idxs]) for d in data]

def freyfaces(distort=False,shuffle=False,ntrain=60000,ntest=10000,onehot=True):
    f = open(os.path.join(datasets_dir,'freyfaces.pkl'),'rb')
    data = cPickle.load(f)
    f.close()

    lenX = len(data) * 0.9

    trX = data[:lenX,:]
    trY = data[:lenX,:1]
    teX = data[lenX:,:]
    teY = data[lenX:,:1]

    return len(trX), trX.astype('float32'),teX.astype('float32'),trY.astype('float32'),teY.astype('float32')


def mnist(binarized = False, distort=False,shuffle=False,shuffle_test_labels=False,ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	if distort:
		ninst = 60000 * 2
		ntrain = ninst
		fd = open(os.path.join(data_dir,'train-images.idx3-ubyte_distorted'))
	else:
		ninst = 60000
		fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((ninst,28*28)).astype(float)

	if distort:
		fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte_distorted'))
	else:
		fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
	
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((ninst))

	fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX /= 255.
	teX /= 255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if shuffle:
		idx = np.random.permutation(ntrain)
		trX_n = trX
		trY_n = trY
		for i in range(ntrain):
			trX[i] = trX_n[idx[i]]
			trY[i] = trY_n[idx[i]]
	
	if shuffle_test_labels:
		idx = np.random.permutation(ntest)
		teY_n = teY
		for i in range(ntest):
			teY[i] = teY_n[idx[i]]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	if binarized:
		#binarized version
		trX = np.random.binomial (n=1, p=trX, size=(trX.shape[0],trX.shape[1]))
		teX = np.random.binomial (n=1, p=teX, size=(teX.shape[0],teX.shape[1]))
	return len(trX), trX.astype('float32'),teX.astype('float32'),trY.astype('float32'),teY.astype('float32')

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not

    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

def visualize (it, images, shape = [30,30], name = 'samples_', p=0):
    image_data = tile_raster_images(images, img_shape=[28,28], tile_shape=shape, tile_spacing=(2,2)) # tile_shape=[numpy.sqrt(len(images)).astype('int32'),len(images)/numpy.sqrt(len(images)).astype('int32')], tile_spacing=(2,2))
    im_new = Image.fromarray(numpy.uint8(image_data))
    im_new.save(name+str(it)+'.png')
    #if (p != 0):
    #    p.terminate()
    #return Popen(['mspaint.exe', 'samples_'+str(it)+'.png']) 


    

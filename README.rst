Theano implementation of the auto-classifiers-encoders (ACE) from "Towards universal neural nets: Gibbs machines and ACE", Galin Georgiev, 
http://arxiv.org/abs/1508.06585

---------------------------
Time spent on GTX 970 GPU:

-non-generative ACE: 	 	      2 sec/epoch for batch size = 10000
-generative ACE:
	-classification task (2): 	  8 sec/epoch for batch size = 10000
	-density estimation task (1): 15 sec/epoch for batch size = 1000
---------------------------
Generative ACE options:

Density estimator (task=1) or classifier (task=2).
Sampling density is either Gaussian (sampling_class=1) or Laplacian (sampling_class=2). 

---------------------------
Generative ACE output for MNIST:

Every 20 epochs, a 30 x 30 matrix of 900 images is saved: 
	-the top 300 images are from the so-called "creative" regime (see sub-section 3.2 in paper). 
	-the middle 300 images are the model reconstructions of the first 300 test images in the "non-creative" regime.
	-the bottom 300 images are the  first 300 test images from the original (possibly binarized)  data set. 

--------------------------
Raw data: 

Obtain the files below from http://yann.lecun.com/exdb/mnist/ and change datasets_dir in toolbox.py to the desired location:

MNIST:
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
Theano implementation of the auto-classifiers-encoders (ACE) from "Towards universal neural nets: Gibbs machines and ACE", Galin Georgiev, GammaDynamics, LLC, 2015.

-------------
The generative ACE can be ran either as a density estimator (task=1) or classifier (task=2).
Its sampling density can be either Gaussian (sampling_class=1) or Laplacian (sampling_class=2). 

The generative ACE generates 30 x 30 matrix of 900 images, on every 20 epochs. The top 300 images are from the so-called "creative" regime (see sub-section 3.2 in paper). 
The middle 300 images are the model reconstructions of the first 300 test images in the "non-creative" regime.
The bottom 300 images are the  first 300 test images from the original (possibly binarized)  data set. 

--------------
Raw data: obtain the files below from http://yann.lecun.com/exdb/mnist/ and change datasets_dir in toolbox.py to the desired location:

MNIST
--------------------------
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
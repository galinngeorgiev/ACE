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

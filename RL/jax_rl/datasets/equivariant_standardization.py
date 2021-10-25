import torch
import torch.nn as nn
import copy
import numpy as np
import objax.nn as nn
import jax
from jax import jit
import jax.numpy as jnp
from emlp.reps import Scalar
from emlp.reps.product_sum_reps import SumRep
import logging
import objax.functional as F
from functools import partial
import objax
from functools import lru_cache as cache

@cache(maxsize=None)
def gate_indices(sumrep): #TODO: add support for mixed_tensors
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    assert isinstance(sumrep,SumRep), f"unexpected type for gate indices {type(sumrep)}"
    channels = sumrep.size()
    perm = sumrep.perm
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rep in sumrep:
        if rep!=Scalar and not rep.is_regular:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars+=1
        i+=rep.size()
    return indices

@cache(maxsize=None)
def scalar_mask(sumrep):
    channels = sumrep.size()
    perm = sumrep.perm
    mask = np.ones(channels)>0
    i=0
    for rep in sumrep:
        if rep!=Scalar: mask[perm[i:i+rep.size()]] = False
        i+=rep.size()
    return mask

@cache(maxsize=None)
def regular_mask(sumrep):
    channels = sumrep.size()
    perm = sumrep.perm
    mask = np.ones(channels)<0
    i=0
    for rep in sumrep:
        if rep.is_regular: mask[perm[i:i+rep.size()]] = True
        i+=rep.size()
    return mask


class TensorBN(nn.BatchNorm0D): #TODO: add suport for mixed tensors.
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep):
        super().__init__(rep.size(),momentum=0.9)
        self.rep=rep
    def __call__(self,x,training): #TODO: support elementwise for regular reps
        #return x #DISABLE BN, harms performance!! !!
        smask = jax.device_put(scalar_mask(self.rep))
        if training:
            m = x.mean(self.redux, keepdims=True)
            v = (x ** 2).mean(self.redux, keepdims=True) - m ** 2
            v = jnp.where(smask,v,ragged_gather_scatter((x ** 2).mean(self.redux),self.rep)) #in non scalar indices, divide by sum squared
            self.running_mean.value += (1 - self.momentum) * (m - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (v - self.running_var.value)
        else:
            m, v = self.running_mean.value, self.running_var.value
        y = jnp.where(smask,self.gamma.value * (x - m) * F.rsqrt(v + self.eps) + self.beta.value,x*F.rsqrt(v+self.eps))#(x-m)*F.rsqrt(v + self.eps))
        return y # switch to or (x-m)

#@export
class EquivStandardizer(object):
    def __init__(self,rep,state_transform,inv_state_transform):
        
        self.state_transform = jit(state_transform)
        self.inv_state_transform = inv_state_transform
        self.mean=  0
        self.pmean =0
        self.var = 0
        self.pvar=0
        self.n = 0
        self.eps = 1e-6
        self.rep,perm = rep.canonicalize()
        invperm = np.argsort(perm)
        the_rep = self.rep
        def ragged_gather_scatter(x):
            x_sorted = x[perm]
            i=0
            y=[]
            for repa, multiplicity in the_rep.reps.items():
                i_end = i+multiplicity*repa.size()
                y.append(x_sorted[i:i_end].reshape(multiplicity,repa.size()).mean(-1,keepdims=True).repeat(repa.size(),axis=-1).reshape(-1))
                i = i_end
            return jnp.concatenate(y)[invperm]
        self.scatter_gather = jit(ragged_gather_scatter)
        self.smask = scalar_mask(self.rep)[invperm]
        self.regular_mask = regular_mask(self.rep)[invperm]

    def add_data(self,x):
        self.n+=1
        tx = self.state_transform(x)
        old_mean = self.mean
        old_pmean = self.pmean
        self.mean = (1-1/self.n)*old_mean + tx/self.n
        self.pmean = jnp.where(self.smask,self.mean,jnp.where(self.regular_mask,self.scatter_gather(self.mean),0*self.mean))
        self.var = (1-1/self.n)*self.var + (tx-old_pmean)*(tx-self.pmean)/self.n
        self.pvar = jnp.where(self.smask,self.var,jnp.where(self.regular_mask,self.scatter_gather(self.var),jnp.ones_like(self.mean)))

    def standardize(self,x):
        return self.inv_state_transform((self.state_transform(x)-self.pmean)*F.rsqrt(self.pvar+self.eps))


# @partial(jit,static_argnums=(1,))
# def ragged_gather_scatter(x,x_rep):
#     y = []
#     i=0
#     for rep in x_rep.reps: # sum -> mean
#         y.append(x[i:i+rep.size()].mean(keepdims=True).repeat(rep.size(),axis=-1))
#         i+=rep.size()
#     return jnp.concatenate(y,-1)

# @partial(jit,static_argnums=(1,))
# def ragged_gather_scatter(x,x_rep):
#     perm = x_rep.argsort()
#     invperm = np.argsort(perm)
#     x_sorted = x[perm]
#     i=0
#     y=[]
#     for rep, multiplicity in x_rep.multiplicities().items():
#         i_end = i+multiplicity*rep.size()
#         y.append(x_sorted[i:i_end].reshape(multiplicity,rep.size()).mean(-1,keepdims=True).repeat(rep.size(),axis=-1).reshape(-1))
#         i = i_end
#     return jnp.concatenate(y)[invperm]
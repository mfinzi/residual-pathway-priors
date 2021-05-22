import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from scipy.special import binom
from jax import jit,vmap
from functools import lru_cache as cache
import jax.numpy as jnp
from jax.experimental.ode import odeint

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)


def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout):
    return Sequential(nn.Linear(cin,cout),swish)#,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,cin,cout,ch=384,num_layers=3):
        super().__init__()
        chs = [cin] + num_layers*[ch]
        logging.info("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,x,training=True):
        return self.net(x)

@export
class DeltaNN(Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,xdim,udim,ch=384,num_layers=3):
        super().__init__()
        self.net = MLP(xdim+udim,xdim,ch,num_layers)
    
    def rollout(self,x0,u,ts):
        xs = [x0]
        norm = 0
        for i in range(len(ts)-1):
            #print(xs[-1].shape,u.shape)
            delta = self.net(jnp.concatenate([xs[-1],u[i]],-1))
            norm += jnp.mean(delta**2,-1)
            xs.append(delta+xs[-1])
        return jnp.stack(xs,0),norm

@export
class NODE(Module,metaclass=Named):
    def __init__(self,xdim,udim,ch=384,num_layers=3):
        super().__init__()
        self.net = MLP(xdim+udim,xdim,ch,num_layers)

    def rollout(self,x0,u,ts,rtol=1e-2):
        ut = lambda t: vmap(jnp.interp,(None,None,1))(t,ts,u)
        def aug_dynamics(z,t):
            F = .3*self.net(jnp.concatenate([z[...,:-1],ut(t)],-1))
            F_norm = (F**2).mean(-1)
            return jnp.concatenate([F,F_norm[...,None]],-1)
        #dynamics = lambda x,t: .3*self.net(jnp.concatenate([x,ut(t)],-1))
        z0 = jnp.concatenate([x0,0*x0[...,:1]])
        zt = odeint(aug_dynamics,z0,ts,rtol=rtol,atol=1e-2)
        xt = zt[...,:-1]
        kinetic = zt[...,-1]
        return xt,kinetic

class SumRollout(Module,metaclass=Named):
    def __init__(self,node,deltann):
        super().__init__()
        self.node=node
        self.deltann=deltann
    def rollout(self,x0,u,ts,rtol=1e-2):
        xt_node,kinetic = self.node.rollout(x0,u,ts,rtol)
        xt_nn,norm = self.deltann.rollout(x0,u,ts)
        return xt_node+.1*xt_nn,.1*norm+kinetic

@export        
def RPPdeltaNode(xdim,udim,ch=384,num_layers=3):
    return SumRollout(NODE(xdim,udim,ch,num_layers),DeltaNN(xdim,udim,ch,num_layers))

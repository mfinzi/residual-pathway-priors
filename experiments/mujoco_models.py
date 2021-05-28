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

    def rollout(self,x0,u,ts,rtol=1e-3):
        ut = lambda t: vmap(jnp.interp,(None,None,1))(t,ts,u)
        def aug_dynamics(z,t):
            F = .3*self.net(jnp.concatenate([z[...,:-1],ut(t)],-1))
            F_norm = (F**2).mean(-1)
            return jnp.concatenate([F,F_norm[...,None]],-1)
        #dynamics = lambda x,t: .3*self.net(jnp.concatenate([x,ut(t)],-1))
        z0 = jnp.concatenate([x0,0*x0[...,:1]])
        zt = odeint(aug_dynamics,z0,ts,rtol=rtol,atol=1e-3)
        xt = zt[...,:-1]
        kinetic = zt[...,-1]
        return xt,kinetic

class SumRollout(Module):
    def __init__(self,node,deltann):
        super().__init__()
        self.node=node
        self.deltann=deltann
    def rollout(self,x0,u,ts,rtol=1e-3):
        xt_node,kinetic = self.node.rollout(x0,u,ts,rtol)
        xt_nn,norm = self.deltann.rollout(x0,u,ts)
        return xt_node/2+xt_nn/2,5*norm+kinetic

@export        
def RPPdeltaNode(xdim,udim,ch=384,num_layers=3):
    return SumRollout(NODE(xdim,udim,ch,num_layers),DeltaNN(xdim,udim,ch,num_layers))

from trainer.hamiltonian_dynamics import hamiltonian_dynamics
from functools import partial

 
# class HMLP(MLP):
#     def H(self,x):
#         return self.net(x).sum()
@export
class HNN(Module,metaclass=Named):
    def __init__(self,xdim,udim,ch=384,num_layers=3):
        super().__init__()
        qdim = xdim//2
        V = MLP(qdim,1,ch,num_layers)
        self.V = V
        L = MLP(qdim,qdim**2,ch,num_layers)
        self.L = L
        self.qdim=qdim
    def tril_Minv(self, q):
        L_q = self.L(q).reshape(self.qdim,self.qdim)
        res = jnp.tril(L_q)
        res = jnp.diag(jax.nn.softplus(1+jnp.diag(res)))-jnp.diag(jnp.diag(res))+res
        return res

    def Minv(self, q, eps=1e-2):
        """Compute the learned inverse mass matrix M^{-1}(q)
        Args:
            q: bs x D Tensor representing the position
        """
        L = self.tril_Minv(q)
        diag_reg = eps*jnp.eye(self.qdim)
        return L@L.T+diag_reg
    
    def M(self,q,v,eps=1e-2):
        #return jnp.linalg.solve(self.Minv(q,eps=eps),v)
        return jnp.linalg.inv(self.Minv(q,eps=eps))@v

    def H(self,z):
        q = z[...,:self.qdim]
        p = z[...,self.qdim:]
        energy = ((self.Minv(q)@p)*p).sum()/2+self.V(q).sum()
        return energy

    def rollout(self,x0,u,ts,rtol=1e-3):
        q0 = x0[:self.qdim]
        v0 = x0[self.qdim:]
        p0 = self.M(q0,v0)
        z0 = jnp.concatenate([q0,p0],axis=-1)
        zt = odeint(partial(hamiltonian_dynamics,self.H),z0,ts,rtol=rtol,atol=1e-3)
        qt = zt[...,:self.qdim]
        pt = zt[...,self.qdim:]
        vt = jnp.squeeze((vmap(self.Minv)(qt)@pt[...,None]),-1)
        return jnp.concatenate([qt,vt],-1),0

@export
class SumRolloutAll(Module):
    def __init__(self,hnn,node,deltann,r2,r3):
        super().__init__()
        self.hnn = hnn
        self.node=node
        self.deltann=deltann
        self.r2=r2
        self.r3=r3
    def rollout(self,x0,u,ts,rtol=1e-3):
        xt_hnn,_ = self.hnn.rollout(x0,u,ts,rtol)
        xt_node,node_kinetic = self.node.rollout(x0,u,ts,rtol)
        xt_nn,nn_norm = self.deltann.rollout(x0,u,ts)
        xout = xt_hnn/3+xt_node/3+xt_nn/3
        regularizer = self.r2*node_kinetic + self.r3*nn_norm
        return xout,regularizer

@export        
def RPPall(xdim,udim,ch=384,num_layers=3,r2=2,r3=4):
    hnn = HNN(xdim,udim,ch,num_layers)
    node = NODE(xdim,udim,ch,num_layers)
    deltann = DeltaNN(xdim,udim,ch,num_layers)
    return SumRolloutAll(hnn,node,deltann,r2,r3)
#%%
import numpy as np
from scipy.linalg import expm
from oil.utils.utils import Named,export
import jax
import jax.numpy as jnp
from emlp.groups import Group


@export
class SL(Group):
    """ The special linear group SL(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = np.zeros((n*n-1,n,n))
        k=0
        for i in range(n):
            for j in range(n):
                if i==j: continue #handle diag elements separately
                self.lie_algebra[k,i,j] = 1
                k+=1
        for l in range(n-1):
            self.lie_algebra[k,l,l] = 1
            self.lie_algebra[k,-1,-1] = -1
            k+=1
        super().__init__(n)
# %%

import jax.numpy as jnp
from emlp.reps import Rep,vis,V,equivariance_error
from emlp.groups import SO,S

class PseudoScalar(Rep):
    is_regular=False
    def __init__(self,G=None):
        self.G=G
        self.concrete = (self.G is not None)
    def __call__(self,G):
        return PseudoScalar(G)
    def size(self):
        return 1
    def __str__(self):
        return "P"
    def rho(self,M):
        sign = jnp.linalg.slogdet(M@jnp.eye(M.shape[0]))[0]
        return sign*jnp.eye(1)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G
    def __hash__(self):
        return hash((type(self),self.G))
    @property
    def T(self):
        return self
from objax.variable import TrainVar
from objax.nn.init import orthogonal
import objax.nn.layers as objax_layers
from objax.module import Module
import objax
import jax.numpy as jnp
import emlp.nn.objax as nn
from emlp.reps import Rep
from oil.utils.utils import Named,export
import logging
import jax

RPP_SCALE=1e-3

class MixedLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, repin, repout):
        nin,nout = repin.size(),repout.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.rep_W = repout<<repin
        
        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        self.Pb = repout.equivariant_projector() # the bias vector has representation repout
        self.Pw = self.rep_W.equivariant_projector()

    def __call__(self, x):
        W = (self.Pw@self.w_equiv.value.reshape(-1)).reshape(*self.w_equiv.value.shape)
        b = self.Pb@self.b.value
        return x@(W.T + self.w_basic.value.T)+b+self.b_basic.value
    
def swish(x):
    return jax.nn.sigmoid(x)*x

class RPPGatedNonlinearity(Module):
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
        self.w_basic = TrainVar(jnp.ones(self.rep.size())*RPP_SCALE)
        
    def __call__(self,values):
        gate_scalars = values[..., nn.gate_indices(self.rep)]
        gated_activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return gated_activations+self.w_basic.value*swish(values[..., :self.rep.size()])

class MixedEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.mixedlinear = MixedLinear(rep_in,nn.gated(rep_out))
        self.bilinear = nn.BiLinear(nn.gated(rep_out),nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self,x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)

@export
class MixedEMLP(Module,metaclass=Named):

  def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[nn.uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else nn.uniform_rep(c,group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        #logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedEMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            MixedLinear(reps[-1],self.rep_out)
        )
  def __call__(self,S,training=True):
      return self.network(S)
    
    
class MixedEMLPH(MixedEMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def H(self,x):#,training=True):
        y = self.network(x)
        return y.sum()
    def __call__(self,x):
        return self.H(x)
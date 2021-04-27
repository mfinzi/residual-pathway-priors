from coax.regularizers import Regularizer
from coax.utils import jit
import jax.numpy as jnp

class RPPRegularizer(Regularizer):
    r"""

    Abstract base class for policy regularizers. Check out
    :class:`coax.regularizers.EntropyRegularizer` for a specific example.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    """
    def __init__(self, f, equiv_wd=1e-3, basic_wd=1e-3):
        self.equiv_wd = equiv_wd
        self.basic_wd = basic_wd
        self.f = f
        
        def function(dist_params, equiv_wd, basic_wd):
#             print(dist_params)
            equiv_l2 = 0.0
            basic_l2 = 0.0
            for k1, v1 in self.f.params.items():
                for k2, v2 in v1.items():
                    if k2.endswith("_basic"):
                        basic_l2 += (v2 ** 2).sum()
                    else:
                        equiv_l2 += (v2 ** 2).sum()
            return (equiv_wd * equiv_l2) + (basic_wd * basic_l2)
            
        
        def metrics(dist_params, equiv_wd, basic_wd):
            equiv_l2 = 0.0
            basic_l2 = 0.0
            for k1, v1 in self.f.params.items():
                for k2, v2 in v1.items():
                    if k2.endswith("_basic"):
                        basic_l2 += (v2 ** 2).sum()
                    else:
                        equiv_l2 += (v2 ** 2).sum()
                        
            return {'RPPRegularizer/equiv_l2':equiv_l2,
                    'RPPRegularizer/basic_l2':basic_l2,
                    'RPPRegularizer/equiv_wd':equiv_wd,
                    'RPPRegularizer/basic_wd':basic_wd,}
            
            
        self._function = jit(function)
        self._metrics_func = jit(metrics)
            
    @property
    def hyperparams(self):
        return {'equiv_wd':self.equiv_wd, 'basic_wd':self.basic_wd}

    @property
    def function(self):
        return self._function

    @property
    def metrics_func(self):
        return self._metrics_func


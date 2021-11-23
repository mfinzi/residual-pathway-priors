from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.networks.common import (MLP, Parameter, Params, PRNGKey,
                                    default_init)
from jax_rl.networks.rpp_emlp_parts import HeadlessRPPEMLP, HeadlessEMLP,parse_rep
from emlp.nn import uniform_rep
from rpp.flax import MixedEMLPBlock,MixedLinear,Sequential,Linear
from emlp.reps import Rep

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
STD_MIN = np.exp(LOG_STD_MIN)
STD_MAX = np.exp(LOG_STD_MAX)

def RPPNormalTanhPolicy(state_rep,action_rep,action_std_rep,G,ch:Sequence[int],
                        state_transform,inv_action_transform,state_dependent_std=True,small_init=True):
    assert state_dependent_std, "only supporting one option for now"
    state_rep = state_rep(G)
    action_rep = action_rep(G)
    action_std_rep = action_std_rep(G) if action_std_rep is not None else action_rep
    body_rpp = HeadlessRPPEMLP(state_rep,G,ch)
    final_rep = parse_rep(ch,G,len(ch))[-1]
    mean_head = MixedLinear(final_rep,action_rep,init_scale=0.01 if small_init else 1.0)
    std_head = MixedLinear(final_rep,action_std_rep,init_scale=0.01 if small_init else 1.0)
    return _RPPNormalTanhPolicy(body_rpp,mean_head,std_head,state_transform,inv_action_transform)

class _RPPNormalTanhPolicy(nn.Module):
    body_rpp:nn.Module
    mean_head:nn.Module
    std_head:nn.Module
    state_transform:callable
    inv_action_transform:callable
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        features = self.body_rpp(self.state_transform(observations))
        means = self.inv_action_transform(self.mean_head(features))
        log_stds = self.inv_action_transform(self.std_head(features))
        #log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = jnp.clip(jax.nn.softplus(log_stds),STD_MIN,STD_MAX)
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=stds * temperature)
        #TODO: figure out how to do the action transform
        return distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), 1))

def EMLPNormalTanhPolicy(state_rep,action_rep,action_std_rep,G,ch:Sequence[int],
                        state_transform,inv_action_transform,state_dependent_std=True,small_init=True):
    assert state_dependent_std, "only supporting one option for now"
    state_rep = state_rep(G)
    action_rep = action_rep(G)
    action_std_rep = action_std_rep(G) if action_std_rep is not None else action_rep
    body_emlp = HeadlessEMLP(state_rep,G,ch)
    final_rep = parse_rep(ch,G,len(ch))[-1]
    mean_head = Linear(final_rep,action_rep,init_scale=0.01 if small_init else 1.0)
    std_head = Linear(final_rep,action_std_rep,init_scale=0.01 if small_init else 1.0)
    return _EMLPNormalTanhPolicy(body_emlp,mean_head,std_head,state_transform,inv_action_transform)

class _EMLPNormalTanhPolicy(nn.Module):
    body_emlp:nn.Module
    mean_head:nn.Module
    std_head:nn.Module
    state_transform:callable
    inv_action_transform:callable
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        features = self.body_emlp(self.state_transform(observations))
        means = self.inv_action_transform(self.mean_head(features))
        log_stds = self.inv_action_transform(self.std_head(features))
        #log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = jnp.clip(jax.nn.softplus(log_stds),STD_MIN,STD_MAX)
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=stds * temperature)
        #TODO: figure out how to do the action transform
        return distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), 1))

    
class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    small_init:bool=True
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)
        scaling = 0.01 if self.small_init else 1.0
        small_init = lambda *args,**kwargs: default_init()(*args,**kwargs)*scaling
        means = nn.Dense(self.action_dim, kernel_init=small_init)(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=small_init)(outputs)
        else:
            log_stds = Parameter(shape=(self.action_dim, ))()

        #log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = jnp.clip(jax.nn.softplus(log_stds),STD_MIN,STD_MAX)
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag= stds * temperature)
        return distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), 1))


    
# def RPPSoftmaxPolicy(state_rep,action_rep,G,ch:Sequence[int],
#                         state_transform,inv_action_transform,state_dependent_std=True):
#     assert state_dependent_std, "only supporting one option for now"
#     state_rep = state_rep(G)
#     action_rep = action_rep(G)
#     body_rpp = HeadlessRPPEMLP(state_rep,G,ch)
#     logits_head = MixedLinear(uniform_rep(ch[-1],G),action_rep)
#     return _RPPSoftmaxPolicy(body_rpp,logits_head,state_transform,inv_action_transform)

# class _RPPSoftmaxPolicy(nn.Module):
#     body_rpp:nn.Module
#     logits_head:nn.Module
#     state_transform:callable
#     inv_action_transform:callable
#     @nn.compact
#     def __call__(self,
#                  observations: jnp.ndarray,
#                  temperature: float = 1.0) -> distrax.Distribution:
#         features = self.body_rpp(self.state_transform(observations))
#         logits = self.inv_action_transform(self.logits_head(features))
#         return distrax.Softmax(logits=logits, temperature=temperature)
    

# class SoftmaxPolicy(nn.Module):
#     hidden_dims: Sequence[int]
#     action_dim: int
#     @nn.compact
#     def __call__(self,
#                  observations: jnp.ndarray,
#                  temperature: float = 1.0) -> distrax.Distribution:
#         logits = MLP(self.hidden_dims, activate_final=False)(observations)
#         return distrax.Softmax(logits=logits, temperature=temperature)

# class NormalTanhMixturePolicy(nn.Module):
#     hidden_dims: Sequence[int]
#     action_dim: int
#     num_components: int = 5

#     @nn.compact
#     def __call__(self,
#                  observations: jnp.ndarray,
#                  temperature: float = 1.0) -> distrax.Distribution:
#         outputs = MLP(self.hidden_dims, activate_final=True)(observations)

#         logits = nn.Dense(self.action_dim * self.num_components,
#                           kernel_init=default_init())(outputs)
#         means = nn.Dense(self.action_dim * self.num_components,
#                          kernel_init=default_init(),
#                          bias_init=nn.initializers.normal(stddev=1.0))(outputs)
#         log_stds = nn.Dense(self.action_dim * self.num_components,
#                             kernel_init=default_init())(outputs)

#         shape = list(observations.shape[:-1]) + [-1, self.num_components]
#         logits = jnp.reshape(logits, shape)
#         mu = jnp.reshape(means, shape)
#         log_stds = jnp.reshape(log_stds, shape)

#         log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

#         components_distribution = distrax.Normal(loc=mu,
#                                                  scale=jnp.exp(log_stds) *
#                                                  temperature)

#         base_dist = distrax.MixtureSameFamily(
#             mixture_distribution=distrax.Categorical(logits=logits),
#             components_distribution=components_distribution)

#         dist = distrax.Transformed(distribution=base_dist,
#                                    bijector=distrax.Tanh())

#         return distrax.Independent(dist, 1)


@jax.partial(jax.jit, static_argnums=1)
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)



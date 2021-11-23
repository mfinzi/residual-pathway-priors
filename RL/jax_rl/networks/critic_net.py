"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jax_rl.networks.common import MLP


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic, -1)

class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions.reshape(observations.shape[0], 1)], -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic, -1)
    
class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims)(observations, actions)
        critic2 = Critic(self.hidden_dims)(observations, actions)
        return critic1, critic2

class DoubleDiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = DiscreteCritic(self.hidden_dims)(observations, actions)
        critic2 = DiscreteCritic(self.hidden_dims)(observations, actions)
        return critic1, critic2    
    


# from jax_rl.networks.rpp_emlp_parts import HeadlessRPPEMLP
from emlp.nn import uniform_rep
from rpp.flax import MixedEMLP,MixedLinear,Sequential
from emlp.reps import Rep,Scalar


def RPPDoubleCritic(state_rep,action_rep,G,ch:Sequence[int],
                        state_transform,action_transform):
    #state_rep = state_rep(G)
    #action_rep = action_rep(G)
    critic1 = MixedEMLP(state_rep+action_rep,Scalar,G,ch)
    critic2 = MixedEMLP(state_rep+action_rep,Scalar,G,ch)
    return _RPPDoubleCritic(critic1,critic2,state_transform,action_transform)

class _RPPDoubleCritic(nn.Module):
    critic1:nn.Module
    critic2:nn.Module
    state_transform:callable
    action_transform:callable
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # TODO add action transform as well as inv action transform
        # Otherwise this won't work with environments that require action transforms
        # like Walker2d. Rember this
        state,actions = self.state_transform(observations),self.action_transform(actions)
        inputs = jnp.concatenate([state, actions], -1)
        c1 = jnp.squeeze(self.critic1(inputs),-1)
        c2 = jnp.squeeze(self.critic2(inputs),-1)
        return c1, c2
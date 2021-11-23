"""Implementations of algorithms for continuous control."""

from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.agents.bc import actor
from jax_rl.datasets import Batch
from jax_rl.networks import policies
from jax_rl.networks.common import InfoDict, Model

_update_jit = jax.jit(actor.update)


class BCLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256)):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = actions.shape[-1]
        actor_def = policies.NormalTanhMixturePolicy(hidden_dims, action_dim)
        self.actor = Model.create(actor_def,
                                  inputs=[actor_key, observations],
                                  tx=optax.adam(learning_rate=actor_lr))
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = policies.sample_actions(self.rng,
                                                    self.actor.apply_fn,
                                                    self.actor.params,
                                                    observations, temperature)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.actor, info = _update_jit(self.actor, batch)
        return info

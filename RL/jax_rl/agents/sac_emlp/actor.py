from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params
import collections

def isDict(pars):
    return isinstance(pars, collections.Mapping)

def get_l2(pars):
    l2 = 0.
    for k, v in pars.items():
        if isDict(v):
            l2 += get_l2(v)            
        elif k.endswith("w"):
            l2 += (v**2).sum()#jnp.linalg.norm(v)**2
                
    return l2

def update(sac: ActorCriticTemp,
           batch: Batch,
          wd) -> Tuple[ActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(sac.rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = sac.actor.apply({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1, q2 = sac.critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * sac.temp() - q).mean()
        
        l2 = get_l2(actor_params)
        actor_loss = actor_loss + wd * l2
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = sac.actor.apply_gradient(actor_loss_fn)

    new_sac = sac.replace(actor=new_actor, rng=rng)

    return new_sac, info

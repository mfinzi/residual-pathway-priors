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
    basic_l2 = 0.
    equiv_l2 = 0.
    for k, v in pars.items():
        if isDict(v):
            sub_basic_l2, sub_equiv_l2 = get_l2(v)
            basic_l2 += sub_basic_l2
            equiv_l2 += sub_equiv_l2
        else:
            if k.endswith("_basic"):
                basic_l2 += (v**2).sum()
            elif k.endswith("_equiv"):
                equiv_l2 += (v**2).sum()
    return basic_l2, equiv_l2

def target_update(sac: ActorCriticTemp, tau: float) -> ActorCriticTemp:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), sac.critic.params,
        sac.target_critic.params)

    new_target_critic = sac.target_critic.replace(params=new_target_params)

    return sac.replace(target_critic=new_target_critic)


def update(sac: ActorCriticTemp, batch: Batch, discount: float,
           soft_critic: bool,cbasic_wd:float,cequiv_wd:float) -> Tuple[ActorCriticTemp, InfoDict]:
    dist = sac.actor(batch.next_observations)
    rng, key = jax.random.split(sac.rng)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q1, next_q2 = sac.target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if soft_critic:
        target_q -= discount * batch.masks * sac.temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = sac.critic.apply({'params': critic_params},
                                  batch.observations, batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        basic_l2, equiv_l2 = get_l2(critic_params)
        closs = critic_loss + cbasic_wd * basic_l2 + cequiv_wd * equiv_l2
        return closs, {
            'critic_loss': closs,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = sac.critic.apply_gradient(critic_loss_fn)

    new_sac = sac.replace(critic=new_critic, rng=rng)

    return new_sac, info

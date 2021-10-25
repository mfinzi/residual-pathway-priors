"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.agents.sac import actor, critic, temperature
from jax_rl.datasets import Batch
from jax_rl.networks import critic_net, policies
from jax_rl.networks.common import InfoDict, Model
from emlp.reps import Rep
from emlp.groups import Group
import collections
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

@jax.partial(jax.jit, static_argnums=(5,))
def _update_jit(sac: ActorCriticTemp, batch: Batch, discount: float,
                tau: float, target_entropy: float,
                update_target: bool,
                 actor_basic_wd: float = 0, actor_equiv_wd: float = 0,
                 critic_basic_wd: float = 0, critic_equiv_wd: float = 0
               ) -> Tuple[ActorCriticTemp, InfoDict]:
    sac, critic_info = critic.update(sac, batch, discount, True,critic_basic_wd,critic_equiv_wd)
    if update_target:
        sac = critic.target_update(sac, tau)
    sac, actor_info = actor.update(sac, batch, actor_basic_wd, actor_equiv_wd)
    sac, alpha_info = temperature.update(sac, actor_info['entropy'],target_entropy)

    return sac, {**critic_info, **actor_info, **alpha_info}

@jax.partial(jax.jit, static_argnums=(5,))
def _update_jit_critic_only(sac: ActorCriticTemp, batch: Batch, discount: float,
                tau: float, target_entropy: float,
                update_target: bool,
                 actor_basic_wd: float = 0, actor_equiv_wd: float = 0,
                 critic_basic_wd: float = 0, critic_equiv_wd: float = 0
               ) -> Tuple[ActorCriticTemp, InfoDict]:
    sac, critic_info = critic.update(sac, batch, discount, True,critic_basic_wd,critic_equiv_wd)
    return sac,critic_info

def clipped_adam(learning_rate,clip_norm=0.5, gan_betas=False):
    if gan_betas:
        b1, b2 = 0.5, 0.999
    else:
        b1, b2 = 0.9, 0.999
    print("B1 = ", b1)
    print("B2 = ", b2)
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.scale_by_adam(b1=b1, b2=b2, eps=1e-8, eps_root=0.0),
        optax.scale(-learning_rate)
    )

class SACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 actor_basic_wd: float = 0.,
                 actor_equiv_wd: float = 0.,
                 
                 critic_lr: float = 3e-4,
                 critic_basic_wd: float = 0.,
                 critic_equiv_wd: float = 0.,

                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 symmetry_group: Optional[Group] = None,
                 state_rep: Optional[Rep] = None,
                 action_rep: Optional[Rep] = None,
                 action_std_rep: Optional[Rep] = None,
                 state_transform=lambda x:x,
                 inv_state_transform=lambda x:x,
                 action_transform=lambda x:x,
                 inv_action_transform=lambda x:x,
                 action_space="continuous",
                 rpp_value=False,
                 rpp_policy=True,
                 small_init=True,
                 middle_rep=None,
                 standardizer=None,
                 clipping=0.5,
                 gan_betas=False):
        self.standardizer= standardizer
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        print("tau = ", tau)
        self.target_update_period = target_update_period
        self.discount = discount
        
        self.actor_basic_wd = actor_basic_wd
        self.actor_equiv_wd = actor_equiv_wd

        self.critic_basic_wd = critic_basic_wd
        self.critic_equiv_wd = critic_equiv_wd
        
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        print("rpp policy is",rpp_policy)
        if rpp_policy: # Use RPP-EMLP policy
            if action_space == "discrete":
                actor_def = policies.RPPSoftmaxPolicy(state_rep,action_rep,symmetry_group,hidden_dims,
                    state_transform=state_transform,inv_action_transform=inv_action_transform)
            else:
                ch = hidden_dims if middle_rep is None else len(hidden_dims)*[middle_rep]
                actor_def = policies.RPPNormalTanhPolicy(state_rep,action_rep,action_std_rep,symmetry_group,ch,
                    state_transform=state_transform,inv_action_transform=inv_action_transform,small_init=small_init)
        else:
            if action_space == "discrete":
                actor_def = policies.SoftmaxPolicy(hidden_dims, action_dim)
            else:
                actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim,small_init=small_init)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=clipped_adam(learning_rate=actor_lr,clip_norm=clipping,
                                            gan_betas=gan_betas))
        if rpp_value:
            ch = hidden_dims if middle_rep is None else len(hidden_dims)*[middle_rep]
            critic_def = critic_net.RPPDoubleCritic(state_rep,action_rep,symmetry_group,ch,
                        state_transform=state_transform,action_transform=action_transform)
        elif action_space=='discrete':
            critic_def = critic_net.DoubleDiscreteCritic(hidden_dims)
        else:
            critic_def = critic_net.DoubleCritic(hidden_dims)
            
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=clipped_adam(learning_rate=critic_lr,clip_norm=clipping,
                                             gan_betas=gan_betas))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=clipped_adam(learning_rate=temp_lr,clip_norm=clipping,gan_betas=gan_betas))

        self.sac = ActorCriticTemp(actor=actor,
                                   critic=critic,
                                   target_critic=target_critic,
                                   temp=temp,
                                   rng=rng)
        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.sac.rng,
                                               self.sac.actor.apply_fn,
                                               self.sac.actor.params,
                                               self.standardizer(observations) if self.standardizer is not None else observations,
                                               temperature)
        self.sac = self.sac.replace(rng=rng)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch,update_policy=True) -> InfoDict:
        'observations', 'actions', 'rewards', 'masks', 'next_observations'
        if self.standardizer is not None:
            standardized_batch = Batch(self.standardizer(batch.observations),batch.actions,
                batch.rewards,batch.masks,self.standardizer(batch.next_observations))
        else:
            standardized_batch = batch
        
        if update_policy:
            self.step += 1
            self.sac, info = _update_jit(
                self.sac, standardized_batch, self.discount, self.tau, self.target_entropy,
                self.step % self.target_update_period == 0,
                self.actor_basic_wd, self.actor_equiv_wd,
                self.critic_basic_wd, self.critic_equiv_wd)
        else:
            self.sac, info = _update_jit_critic_only(
                self.sac, standardized_batch, self.discount, self.tau, self.target_entropy,
                self.step % self.target_update_period == 0,
                self.actor_basic_wd, self.actor_equiv_wd,
                self.critic_basic_wd, self.critic_equiv_wd)
        return info

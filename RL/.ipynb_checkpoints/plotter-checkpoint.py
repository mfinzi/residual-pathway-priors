import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from coax.value_losses import mse
import matplotlib.pyplot as plt
from matplotlib import animation

from emlp import T, Scalar
from emlp.groups import SO, S, O, Trivial,Z
from emlp_haiku import EMLPBlock, Sequential, Linear,EMLP, MLP
from emlp.reps import Rep
from emlp.nn import gated,gate_indices,uniform_rep
from math import prod
from representations import PseudoScalar
from mixed_emlp_haiku import MixedEMLP


# the name of this script
name = 'a2c'

# the cart-pole MDP
# env = gym.make('CartPole-v0')
env = gym.make("rpp_gym:InclinedCartpole-v0")
# env.alpha=0.8

group=Z(2)
rep_in = PseudoScalar()*prod(env.observation_space.shape)
rep_out = T(1)#*env.action_space.n#prod(env.action_space.shape)
nn_pi = EMLP(rep_in,rep_out,group,ch=300,num_layers=3)

def func_pi(S, is_training):
    return {'logits': nn_pi(S)}

pi = coax.Policy(func_pi, env)
pi.params = coax.utils.load("./emlp_pi_params (1).lz4")



def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

s = env.reset()
frames = []
for _ in range(100):
    frames.append(env.render(mode='rgb_array'))
    
    a = pi(s)
    s, r, done, info = env.step(a)

#     s, r, done, info = env.step(env.action_space.sample())

    if done:
        break
env.close()
save_frames_as_gif(frames)


import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from coax.value_losses import mse

import matplotlib.pyplot as plt



def main(args):
    # the name of this script
    name = 'a2c'

    # the cart-pole MDP
    # env = gym.make('CartPole-v0')
    env = gym.make("rpp_gym:InclinedCartpole-v0")
    env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="windy pendulum ablation")
    parser.add_argument( 
        "--network",
        type=str,
        default="EMLP",
        help="type of network {EMLP, MixedEMLP, MLP}",
    )
    parser.add_argument(
        "--equiv_wd",
        type=float,
        default=1e-4,
        help="basic weight decay",
    )
    args = parser.parse_args()

    main(args)
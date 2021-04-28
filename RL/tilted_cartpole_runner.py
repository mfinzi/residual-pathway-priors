import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from coax.value_losses import mse
import argparse
import pickle

from emlp import T, Scalar
from emlp.groups import SO, S, O, Trivial,Z
import emlp.nn.haiku as ehk
from emlp.reps import Rep
from emlp.nn import gated,gate_indices,uniform_rep
from math import prod
from representations import PseudoScalar
from mixed_emlp_haiku import MixedEMLP
from RPPRegularizer import RPPRegularizer

def main(args):
    # the name of this script
    name = 'a2c'

    # the cart-pole MDP
    # env = gym.make('CartPole-v0')
    env = gym.make("rpp_gym:InclinedCartpole-v0")
    env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")
    if args.alpha is not None:
        env.alpha = args.alpha
        
    group=Z(2)
    rep_in = PseudoScalar()*prod(env.observation_space.shape)
    rep_out = T(1)#*env.action_space.n#prod(env.action_space.shape)
    
    if args.network.lower() == "emlp":
        nn_pi = ehk.EMLP(rep_in,rep_out,group,ch=100,num_layers=2)
        nn_v = ehk.EMLP(rep_in,T(0),group,ch=100,num_layers=2)
    elif args.network.lower() == 'mixedemlp':
        nn_pi = MixedEMLP(rep_in,rep_out(group),group,ch=100,num_layers=2)
        nn_v = MixedEMLP(rep_in,T(0),group,ch=100,num_layers=2)
    else:
        nn_pi = ehk.MLP(rep_in,rep_out(group),group,ch=100,num_layers=2)
        nn_v = ehk.MLP(rep_in,T(0),group,ch=100,num_layers=2)
        
    def func_pi(S, is_training):
        return {'logits': nn_pi(S)}

    def func_v(S, is_training):
        return nn_v(S).reshape(-1)
    

    # these optimizers collect batches of grads before applying updates
    optimizer_v = optax.chain(optax.apply_every(k=32), optax.adam(0.002))
    optimizer_pi = optax.chain(optax.apply_every(k=32), optax.adam(0.001))

    # value function and its derived policy
    v = coax.V(func_v, env)
    pi = coax.Policy(func_pi, env)

    # policy regularization
    pi_regularizer = RPPRegularizer(pi, basic_wd=args.basic_wd, equiv_wd=args.equiv_wd)
    
    # experience tracer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)

    # updaters
    vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optimizer_pi,
                                                  regularizer=pi_regularizer)
    simple_td = coax.td_learning.SimpleTD(v, loss_function=mse, optimizer=optimizer_v)

    epoch_rewards = []

    # train
    for ep in range(1000):
        s = env.reset()
        er = 0
        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            if done and (t == env.spec.max_episode_steps - 1):
                r = 1 / (1 - tracer.gamma)
            er+=r
            tracer.add(s, a, r, done)
            while tracer:
                transition_batch = tracer.pop()
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = vanilla_pg.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

            if done:
                break

            s = s_next

        print("Epoch reward",er)
        epoch_rewards.append(er)
        # early stopping
        if env.avg_G > env.spec.reward_threshold:
            break
            
            
    fname = args.network + "_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd)
    
    filehandler = open("./saved-outputs/" + fname + "epoch_rewards.pkl","wb")
    pickle.dump(epoch_rewards,filehandler)
    filehandler.close()

    coax.utils.dump(pi.params, "./saved-outputs/" + fname + "_pi_params.lz4")
    coax.utils.dump(v.params, "./saved-outputs/" + fname + "_v_params.lz4")
    

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
        help="equiv weight decay",
    )
    parser.add_argument(
        "--basic_wd",
        type=float,
        default=1e-4,
        help="basic weight decay",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="option to change angle of cartpole env",
    )
    args = parser.parse_args()

    main(args)
import os
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import sys
sys.path.append("../")
from jax_rl.agents import AWACLearner, SACLearner,SACEMLPLearner
from jax_rl.datasets import ReplayBuffer
from jax_rl.evaluation import evaluate,rpp_evaluate
from jax_rl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Ant-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/emlp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('rpp_value', False, 'Also use RPP for value function')
flags.DEFINE_string('group', '', 'Also use RPP for value function')
flags.DEFINE_float('wd', 1e-6, 'Basic weight decay')
flags.DEFINE_list('hidden_dims', [265,256], 'Dimension of hidden layers')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

from representations import environment_symmetries
from emlp.groups import *
from jax import jit,vmap

def main(_):
    print("CWD = ", os.getcwd())
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    kwargs.update(environment_symmetries[FLAGS.env_name])
    
    hidden_dims = tuple(int(hd) for hd in FLAGS.hidden_dims)
    kwargs['hidden_dims'] = hidden_dims
    
    kwargs['rpp_value']=FLAGS.rpp_value
    if FLAGS.group:
        kwargs['symmetry_group']=eval(FLAGS.group)
    kwargs['state_rep'] = kwargs['state_rep'](kwargs['symmetry_group'])
    kwargs['action_rep'] = kwargs['action_rep'](kwargs['symmetry_group'])
    algo = kwargs.pop('algo')
    assert algo=='sac', "other RL algos not yet supported"
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACEMLPLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                            np.asarray(env.action_space.sample())[None], 
                           actor_wd=FLAGS.wd,**kwargs)
        policy_mean_fn = jit(lambda p,x: agent.sac.actor.apply_fn.apply({'params':p},x)._distribution._loc)
    else:
        raise NotImplementedError()

    @jit
    def reprhos(x):
        gs = kwargs['symmetry_group'].samples(x.shape[0])
        ring = vmap(kwargs['state_rep'].rho_dense)(gs)
        routg = vmap(kwargs['action_rep'].rho_dense)(gs)
        return ring,routg

    action_dim = env.action_space.shape[0] if kwargs['action_space']=='continuous' else 1
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            if (i//FLAGS.eval_interval)%4==0: # only do equivariance calc ever 4 evals
                eval_stats = rpp_evaluate(agent,policy_mean_fn, eval_env, FLAGS.eval_episodes,kwargs,reprhos)
            else:
                eval_stats = evaluate(agent,eval_env,FLAGS.eval_episodes)
            
            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()


            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)

import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
from emlp.reps import Scalar,Vector,T
from torch.utils.data import Dataset
from oil.utils.utils import export,Named,Expression,FixedNumpySeed
from emlp.groups import SO,O,Trivial,Lorentz,RubiksCube,Cube
from functools import partial
import itertools
from jax import vmap,jit
from objax import Module
import os,glob,argparse
import argparse
import numpy as np
import pandas as pd

@export
class ModifiedInertia(Dataset,metaclass=Named):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N,self.dim)
        self.X[:,:k] = F.softplus(self.X[:,:k]) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        g = I[2]# z axis
        v = (inertia*g).sum(-1)
        vgT = v[:,:,None]*g[None,None,:]
        target = inertia + 3e-1*vgT
        self.Y = target.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats =0,1,0,1#Xmean,Xstd,Ymean,Ystd

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]


@export
class Inertia(Dataset):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N,self.dim)
        self.X[:,:k] = F.softplus(self.X[:,:k]) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        self.Y = inertia.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        # One has to be careful computing offset and scale in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:,:k]).mean(0)#.std(0)
        #Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:,k:].reshape(N,k,3)).mean((0,2))[:,None] + np.zeros((k,3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        #Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0,1)) + np.zeros_like(Ymean)
        self.stats =0,1,0,1#Xmean,Xstd,Ymean,Ystd

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug(self,model):
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)

timestep_table = {'Humanoid-v2':0.015,'Walker2d-v2':.008,'Ant-v2':.05,
                    'Swimmer-v2':.04,'Hopper-v2':.008,'HalfCheetah-v2':.05,
                    'HopperFull-v0':.08}
statedim_table = {'Humanoid-v2':45,'Walker2d-v2':16,'Ant-v2':27,
                    'Swimmer-v2':8,'Hopper-v2':11,'HalfCheetah-v2':17,
                    'HopperFull-v0':12}

import scipy
import scipy.ndimage

@export
class MujocoRegression(Dataset,metaclass=Named):
    def __init__(self,N=10000,env='Humanoid-v2',chunk_len=5,seed=0):
        super().__init__()
        self.X = np.load(f"{env}_cl{chunk_len}_xdata.npy") #(N,chunk_len,d_obs)
        self.U = np.load(f"{env}_cl{chunk_len}_udata.npy") #(N,chunk_len,d_action)
        # Subsample to size:
        with FixedNumpySeed(seed):
            ids = np.random.choice(self.X.shape[0], N//chunk_len, replace=False)
        self.X = scipy.ndimage.gaussian_filter1d(self.X[ids,:,:statedim_table[env]],2,axis=1)
        #self.X/=self.X.std((0,1))
        self.U = scipy.ndimage.gaussian_filter1d(self.U[ids],2,axis=1)
        #self.U/=self.U.std((0,1))
        self.xdim = self.X.shape[-1]
        self.udim = self.U.shape[-1]
        self.T = np.arange(chunk_len)*timestep_table[env]

    def __getitem__(self,i):
        return (self.X[i,0], self.U[i,:],self.T), self.X[i]
    def __len__(self):
        return self.X.shape[0]


@export
class MujocoRollouts(Dataset,metaclass=Named):
    def __init__(self,N=100,env='Humanoid-v2',seed=0):
        super().__init__()
        self.X = np.load(f"{env}_episodes_xdata.npy") #(N,traj_len,d_obs)
        self.U = np.load(f"{env}_episodes_udata.npy") #(N,traj_len,d_action)
        # Subsample to size:
        with FixedNumpySeed(seed):
            ids = np.random.choice(self.X.shape[0], N, replace=False)
        self.X = scipy.ndimage.gaussian_filter1d(self.X[ids,:,:statedim_table[env]],2,axis=1)
        #self.X/=self.X.std((0,1))
        self.U = scipy.ndimage.gaussian_filter1d(self.U[ids],2,axis=1)
        #self.U/=self.U.std((0,1))
        self.xdim = self.X.shape[-1]
        self.udim = self.U.shape[-1]
        self.T = np.arange(self.X.shape[1])*timestep_table[env]

    def __getitem__(self,i):
        return (self.X[i,0], self.U[i,:],self.T), self.X[i]
    def __len__(self):
        return self.X.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Mujoco ReplayBuffer into Offline dynamics dataset')
    parser.add_argument('--path', type=str, default=os.path.expanduser("~/rpp-rl"), help='folder containing csvs')
    parser.add_argument('--chunk_len', type=int, default=5, help="size of training trajectory chunks")
    parser.add_argument('--rollout_len', type=int, default=200, help="max size of test rollouts")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    args = parser.parse_args()
    pths = glob.glob(args.path+"/*.csv")
    print(f"Found replay buffers: {pths}")
    with FixedNumpySeed(args.seed):
        for pth in pths:
            envname = pth.split('_')[-1].split('.')[0]
            df = pd.read_csv(pth)

            all_states = df[[colname for colname in df.columns if colname[0]=='x']]
            all_controls = df[[colname for colname in df.columns if colname[0]=='u']]

            episode_states = np.split(all_states, np.where(df['restarts'])[0])
            episode_controls = np.split(all_controls, np.where(df['restarts'])[0])

            test_traj_x = []
            test_traj_u = []
            x_chunks = []
            u_chunks = []
            for n,(states, controls) in enumerate(zip(episode_states,episode_controls)):
                if n%10==0: # separate out 10% for full rollouts
                    i_start = np.random.randint(100)
                    xs = states.values[i_start:i_start+args.rollout_len]
                    if len(xs)==args.rollout_len:
                        test_traj_x.append(xs)
                        test_traj_u.append(controls.values[i_start:i_start+args.rollout_len])
                        continue
                i_start = np.random.randint(args.chunk_len)
                chunk_x = states.values[i_start:i_start+args.chunk_len*((len(states)-i_start)//args.chunk_len)]
                x_chunks.append(chunk_x.reshape(-1,args.chunk_len,chunk_x.shape[-1]))
                chunk_u = controls.values[i_start:i_start+args.chunk_len*((len(controls)-i_start)//args.chunk_len)]
                u_chunks.append(chunk_u.reshape(-1,args.chunk_len,chunk_u.shape[-1]))
            x_chunks = np.concatenate(x_chunks,axis=0)
            u_chunks = np.concatenate(u_chunks,axis=0)
            print(f"Found {x_chunks.shape} chunks from {len(episode_states)} episodes on {envname}")
            #if test_traj_x:
            np.save(f"{envname}_cl{args.chunk_len}_xdata.npy",x_chunks)
            np.save(f"{envname}_cl{args.chunk_len}_udata.npy",u_chunks)
            print(f"Saved {len(test_traj_x)} test episodes.")
            np.save(f"{envname}_episodes_xdata.npy",np.stack(test_traj_x,axis=0))
            np.save(f"{envname}_episodes_udata.npy",np.stack(test_traj_u,axis=0))



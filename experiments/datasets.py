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
class MujocoRegression(Dataset,metaclass=Named):
    def __init__(self,N=10000,env='Humanoid-v2',chunk_len=5):
        super().__init__()
        self.X = np.load(f"{env}_cl{chunk_len}_xdata.npy") #(N,chunk_len,d_obs)
        self.U = np.load(f"{env}_cl{chunk_len}_udata.npy") #(N,chunk_len,d_action)
        # Subsample to size:
        ids = np.random.choice(self.X.shape[0],N//chunk_len,replace=False)
        self.X = self.X[ids]
        self.U = self.U[ids]
    
    def __getitem__(self,i):
        return (self.X[i,0],self.U[i,:]), self.X[i]
    def __len__(self):
        return self.X.shape[0]
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert Mujoco ReplayBuffer into Offline dynamics dataset')
    parser.add_argument('--path', type=str, default=os.path.expanduser("~/rpp-rl"),help='folder containing csvs')
    parser.add_argument('--chunk_len',type=int,default=5,help="size of training trajectory chunks")
    args = parser.parse_args()
    pths =glob.glob(args.path+"/*.csv")
    print(f"Found replay buffers: {pths}")
    for pth in pths:
        envname = pth.split('_')[-1].split('.')[0]
        df = pd.read_csv(pth)

        all_states = df[[colname for colname in df.columns if colname[0]=='x']]
        all_controls = df[[colname for colname in df.columns if colname[0]=='u']]

        episode_states = np.split(all_states,np.where(df['restarts'])[0])
        episode_controls = np.split(all_controls,np.where(df['restarts'])[0])

        x_chunks = []
        u_chunks = []
        for states,controls in zip(episode_states,episode_controls):
            i_start = np.random.randint(args.chunk_len)
            chunk_x = states.values[i_start:i_start+args.chunk_len*((len(states)-i_start)//args.chunk_len)]
            x_chunks.append(chunk_x.reshape(-1,args.chunk_len,chunk_x.shape[-1]))
            chunk_u = controls.values[i_start:i_start+args.chunk_len*((len(controls)-i_start)//args.chunk_len)]
            u_chunks.append(chunk_u.reshape(-1,args.chunk_len,chunk_u.shape[-1]))
        x_chunks = np.concatenate(x_chunks,axis=0)
        u_chunks = np.concatenate(u_chunks,axis=0)
        np.save(f"{envname}_cl{args.chunk_len}_xdata.npy",x_chunks)
        np.save(f"{envname}_cl{args.chunk_len}_udata.npy",u_chunks)


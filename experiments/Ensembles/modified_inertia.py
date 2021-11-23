from emlp.nn import MLP,EMLP,MLPH,EMLPH
from emlp.groups import SO2eR3,O2eR3,DkeR3,Trivial, SO
from emlp.reps import Scalar
import sys
sys.path.append("../trainer/")
from hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnn_trial
from hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from rpp.objax import MixedEMLP, MixedEMLPH
sys.path.append("../")
from datasets import ModifiedInertia, Inertia
import torch.nn as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import objax
from jax import vmap
import argparse
from tqdm import tqdm
import os



def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

def scale_adjusted_rel_err(a,b,g):
    return  jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean())+jnp.abs(g-jnp.eye(g.shape[-1])).mean())

def equivariance_err(model,mb,group=None):
    x,y = mb
    group = model.G if group is None else group
    gs = group.samples(x.shape[0])
    rho_gin = vmap(model.rep_in.rho_dense)(gs)
    rho_gout = vmap(model.rep_out.rho_dense)(gs)
    y1 = model((rho_gin@x[...,None])[...,0])
    y2 = (rho_gout@model(x)[...,None])[...,0]
    return np.asarray(scale_adjusted_rel_err(y1,y2,gs))

def mse(mdl, x, y):
    yhat = mdl(x)
    return ((yhat-y)**2).mean()

def main(args):

    num_epochs=50
    ndata=1000+2000
    seed=2021
    
    lr = 3e-3

    bs = 500
    logger = []
    if args.network.lower() == 'mixedemlp':
        savedir = "./saved-outputs/inertia_basic" + str (args.basic_wd) + "_equiv" + str(args.equiv_wd) + "/"
        
    elif args.network.lower() == 'emlp':
        savedir = "./saved-outputs/inertia_emlp/"
    else:
        savedir = "./saved-outputs/inertia_mlp/"
        
    os.makedirs(savedir, exist_ok=True)
    
    for trial in range(10):
        
        if args.modified.lower() == "t":
            dset = ModifiedInertia(3000) # Initialize dataset with 1000 examples
            dataname = 'modifiedinertia'
        else:
            dset = Inertia(3000)
            dataname = 'inertia'
        split={'train':-1,'val':1000,'test':1000}
        datasets = split_dataset(dset,splits=split)
        dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
        trainloader = dataloaders['train']
        testloader = dataloaders['test']

        G = dset.symmetry
        if args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out, group=G,num_layers=3,ch=384)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLP(dset.rep_in, dset.rep_out, group=G,num_layers=3,ch=384)
        else:
            model = MLP(dset.rep_in, dset.rep_out, group=G,num_layers=3,ch=384)
        opt = objax.optimizer.Adam(model.vars())#,beta2=.99)


        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            return ((yhat-y)**2).mean()

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            basic_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('w_basic'))
            equiv_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('w_equiv'))

            return mse + (args.basic_wd*basic_l2) + (args.equiv_wd*equiv_l2)

        grad_and_val = objax.GradValues(loss, model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr):
            g, v = grad_and_val(x, y)
            opt(lr=lr, grads=g)
            return v

        for epoch in tqdm(range(num_epochs)):
            train_mse = np.mean([train_op(jnp.array(x),jnp.array(y),lr) for (x,y) in trainloader])
            if (epoch % args.save_every == 0):
#                 fname = dataname + "_model_epoch" + str(epoch)+ "_trial" + str(trial) + ".npz"
#                 objax.io.save_var_collection(savedir + fname, model.vars())
                

                train_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in trainloader])
                test_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
                equiv_err = np.mean([equivariance_err(model, mb) for mb in testloader])
                logger.append([trial, epoch, train_mse, test_mse, equiv_err])

                
        train_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in trainloader])
        test_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
        equiv_err = np.mean([equivariance_err(model, mb) for mb in testloader])
        logger.append([trial, epoch, train_mse, test_mse, equiv_err])
        
#         fname = dataname + "_model_epoch" + str(epoch)+ "_trial" + str(trial) + ".npz"
#         objax.io.save_var_collection(savedir + fname, model.vars())

    save_df = pd.DataFrame(logger)
    save_df.to_pickle(savedir + dataname + "_loss_log.pkl")
    

    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument( 
        "--basic_wd",
        type=float,
        default=1.,
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=float,
        default=1e-4,
        help="equiv weight decay",
    )
    parser.add_argument( 
        "--network",
        type=str,
        default="MixedEMLP",
        help="type of network {EMLP, MixedEMLP, MLP}",
    )
    parser.add_argument( 
        "--modified",
        type=str,
        default="t",
        help="{T,F} flag for modified vs plain inertia",
    )
    parser.add_argument( 
        "--save_every",
        type=int,
        default="5",
        help="save every n epochs",
    )
    args = parser.parse_args()

    main(args)
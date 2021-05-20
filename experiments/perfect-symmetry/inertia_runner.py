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
from datasets import Inertia
import torch.nn as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import objax
import argparse
from tqdm import tqdm
import os

def main(args):

    num_epochs=300
    ndata=1000+2000
    seed=2021
    
    lr = 3e-3

    bs = 500
    logger = []
    
    for trial in range(10):
        

        dset = Inertia(3000) # Initialize dataset with 1000 examples
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

        train_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in trainloader])
        test_mse = np.mean([mse(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
        logger.append([trial, train_mse, test_mse])
        if args.network.lower() != 'mixedemlp':
            fname = "mdl_inertia_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) +\
                    "_trial" + str(trial) + ".npz"
            objax.io.save_var_collection("./saved-outputs/" + fname, model.vars())

    save_df = pd.DataFrame(logger)
    fname = "inertia_log_" + args.network + "_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) + ".pkl"
    os.makedirs("./saved-outputs/",exist_ok=True)
    save_df.to_pickle("./saved-outputs/" + fname)
    

    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument( 
        "--basic_wd",
        type=float,
        default=1e2,
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=float,
        default=.001,
        help="equiv weight decay",
    )
    parser.add_argument( 
        "--network",
        type=str,
        default="MixedEMLP",
        help="type of network {EMLP, MixedEMLP, MLP}",
    )
    args = parser.parse_args()

    main(args)
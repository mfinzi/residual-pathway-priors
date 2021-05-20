from emlp.nn import MLP,EMLP,MLPH,EMLPH
from emlp.groups import SO2eR3,O2eR3,DkeR3,Trivial
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
from datasets import ModifiedInertia
import torch.nn as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import objax
import argparse
from tqdm import tqdm

def main(args):

    num_epochs=1000
    ndata=5000
    seed=2021


    lr = 3e-3

    split={'train':500,'val':.1,'test':.1}
    bs = 500
    logger = []
    for trial in range(10):
        dataset = DoubleSpringPendulum
        base_ds = dataset(n_systems=ndata,chunk_len=5)
        datasets = split_dataset(base_ds,splits=split)

        dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                                             num_workers=0,pin_memory=False)) for k,v in datasets.items()}
        trainloader = dataloaders['train'] 
        testloader = dataloaders['val'] 

        net_config={'num_layers':3,'ch':128,'group':base_ds.symmetry}
        if args.network.lower() == "emlp":
            model = EMLPH(base_ds.rep_in, Scalar, **net_config)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLPH(base_ds.rep_in, Scalar, **net_config)
        else:
            model = MLPH(base_ds.rep_in, Scalar, **net_config)

        opt = objax.optimizer.Adam(model.vars())

        lr = 3e-3

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def mse(minibatch):
            (z0, ts), true_zs = minibatch
            pred_zs = BHamiltonianFlow(model,z0,ts[0])
            return jnp.mean((pred_zs - true_zs)**2)

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(minibatch):
            """ Standard cross-entropy loss """
            (z0, ts), true_zs = minibatch
            pred_zs = BHamiltonianFlow(model,z0,ts[0])
            mse = jnp.mean((pred_zs - true_zs)**2)


            basic_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('_basic'))
            equiv_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if not k.endswith('_basic'))
            return mse + (args.basic_wd*basic_l2) + (args.equiv_wd*equiv_l2)

        grad_and_val = objax.GradValues(loss, model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(batch, lr):
            g, v = grad_and_val(batch)
            opt(lr=lr, grads=g)
            return v



        for epoch in tqdm(range(num_epochs)):
            tr_loss_wd = np.mean([train_op(batch,lr) for batch in trainloader])
            
        test_loss = np.mean([mse(batch) for batch in testloader])
        tr_loss = np.mean([mse(batch) for batch in trainloader])
        logger.append([trial, tr_loss, test_loss])
        if args.network.lower() != "mixedemlp":
            fname = "log_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) +\
                    "_trial" + str(trial) + ".npz"
            objax.io.save_var_collection("./saved-outputs/" + fname, model.vars())
        
    save_df = pd.DataFrame(logger)
    fname = "log_" + args.network + "_basic" + str(args.basic_wd) +\
            "_equiv" + str(args.equiv_wd) + ".pkl"
    save_df.to_pickle("./saved-outputs/" + fname)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="pendulum ablation")
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
        help="basic weight decay",
    )
    parser.add_argument( 
        "--network",
        type=str,
        default="MixedEMLP",
        help="type of network {EMLP, MixedEMLP, MLP}",
    )
    args = parser.parse_args()

    main(args)
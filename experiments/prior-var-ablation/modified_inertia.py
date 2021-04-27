from emlp.nn import MLP,EMLP,MLPH,EMLPH
from emlp.groups import SO2eR3,O2eR3,DkeR3,Trivial
from emlp.reps import Scalar
from trainer.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnn_trial
from trainer.hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from soft_emlp import MixedEMLP, MixedEMLPH
from datasets import ModifiedInertia
import torch.nn as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import objax
import argparse
from tqdm import tqdm
import os

def main(args):

    num_epochs=1000
    ndata=1000+2000
    seed=2021
    
    lr = 3e-3

    split={'train':-1,'val':1000,'test':1000}
    bs = 500
    
    dataset = ModifiedInertia
    
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(ndata)
        datasets = split_dataset(base_ds,splits=split)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                                         num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    trainloader = dataloaders['train'] 
    testloader = dataloaders['val'] 

    net_config={'num_layers':3,'ch':384,'group':base_ds.symmetry}
    model = MixedEMLP(base_ds.rep_in, base_ds.rep_out, **net_config)

    opt = objax.optimizer.Adam(model.vars())

    lr = 3e-3

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(minibatch):
        """ l2 regularized MSE """
        x,y = minibatch
        mse = jnp.mean((model(x)-y)**2)

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

    logger = []
    for epoch in tqdm(range(num_epochs)):
        tr_loss = np.mean([train_op(batch,lr) for batch in trainloader])
        test_loss = None
        if not epoch%10:
            test_loss = np.mean([loss(batch) for batch in testloader])
            print(f"train loss: {tr_loss} test loss: {test_loss}")

        logger.append([epoch, tr_loss, test_loss])

    save_df = pd.DataFrame(logger)
    print(f"Outcome:\n{save_df.iloc[-10:]}")
    fname = "inertia_log_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) + ".pkl"
    os.makedirs("./saved-outputs/",exist_ok=True)
    save_df.to_pickle("./saved-outputs/" + fname)
    
    fname = "inertia_mdl_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) + ".npz"
    objax.io.save_var_collection("./saved-outputs/" + fname, model.vars())

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
        default=0*1e-6,
        help="basic weight decay",
    )
    args = parser.parse_args()

    main(args)
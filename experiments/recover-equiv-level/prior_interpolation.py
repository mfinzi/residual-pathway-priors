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

    dataset = WindyDoubleSpringPendulum
    base_ds = dataset(wind_scale=1e-2, n_systems=ndata,chunk_len=5)
    datasets = split_dataset(base_ds,splits=split)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                                         num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    trainloader = dataloaders['train'] 
    testloader = dataloaders['val'] 

    net_config={'num_layers':3,'ch':128,'group':base_ds.symmetry}
    model = MixedEMLPH(base_ds.rep_in, Scalar, **net_config)

    opt = objax.optimizer.Adam(model.vars())

    lr = 3e-3
    
    basic_wd = args.base_l2 * 1./(np.maximum(args.alpha, 1e-6))
    equiv_wd = args.base_l2 * 1./(np.maximum((1 - args.alpha), 1e-6))

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


        basic_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('w_basic'))
        equiv_l2 = sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('w'))
        return mse + (basic_wd * basic_l2) + (equiv_wd * equiv_l2)

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(batch, lr):
        g, v = grad_and_val(batch)
        opt(lr=lr, grads=g)
        return v



    for epoch in tqdm(range(num_epochs)):
        tr_loss_wd = np.mean([train_op(batch,lr) for batch in trainloader])
        test_loss = tr_loss = None
        if not epoch%10:
            test_loss = np.mean([mse(batch) for batch in testloader])
            tr_loss = np.mean([mse(batch) for batch in trainloader])

        logger.append([epoch, tr_loss, test_loss])
        
    test_loss = np.mean([mse(batch) for batch in testloader])
    tr_loss = np.mean([mse(batch) for batch in trainloader])
    logger.append([epoch, tr_loss, test_loss])

    save_df = pd.DataFrame(logger)
    fname = "log_alpha" + str(args.alpha) + "_trial" + str(args.trial) + ".pkl"
    save_df.to_pickle("./saved-outputs/" + fname)
    
    fname = "mdl_alpha" + str(args.alpha) + "_trial" + str(args.trial) + ".npz"
    objax.io.save_var_collection("./saved-outputs/" + fname, model.vars())
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="windy pendulum ablation")
    parser.add_argument(
        "--base_l2",
        type=float,
        default=1e-5,
        
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.0,
        help='ratio of equiv to basic l2'
    )
    
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="just an identifying tag for ",
    )
    
    args = parser.parse_args()

    main(args)
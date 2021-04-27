
import jax.numpy as jnp
import jax
import itertools
import numpy as np
import torch
import objax
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
import logging
from emlp.nn import Standardize
from tagging.pointconv_base import ResNet
from tagging.tagging_dataset import TopTagging,collate_fn
from trainer.tagging_trainer import TaggingTrainer

def makeTrainer(*,network=ResNet,num_epochs=5,seed=2021,aug=False,
                bs=30,lr=1e-3,split={'train':100000,'val':10000},
                net_config={'k':256,'num_layers':4},
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.2}},save=False):
    # Prep the datasets splits, model, and dataloaders
    datasets = {key:TopTagging(split=key) for key in ['train','val']}
    print({key:len(ds) for key,ds in datasets.items()})
    datasets = {key:split_dataset(ds,{key:split[key]})[key] for key,ds in datasets.items()}
    model = network(4,2,**net_config)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False,collate_fn=collate_fn,drop_last=True)) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],0,None,10) #for logging subsample dataset by 10x
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: 1#lr*cosLr(num_epochs)(e)
    return TaggingTrainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    Trial(argupdated_config(makeTrainer.__kwdefaults__))
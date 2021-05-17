from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, FixedNumpySeed,FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import emlp.nn
import emlp.groups
import objax
from trainer.hamiltonian_dynamics import DoubleSpringPendulum
from jax import vmap,jit
from functools import partial
import jax.numpy as jnp
import numpy as np

levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                    'warn': logging.WARNING,'warning': logging.WARNING,
                    'info': logging.INFO,'debug': logging.DEBUG}

from trainer.classifier import Regressor,Classifier

def _rollout(model,x0,u,ts):
    return model.rollout(x0,u,ts)

def rollout(model):
    return jit(vmap(partial(_rollout,model),(0,0,None)))

class RolloutTrainer(Regressor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.rollout = rollout(model)
        self.loss = objax.Jit(self.loss,model.vars())
        #self.model = objax.Jit(self.model)
        self.gradvals = objax.Jit(objax.GradValues(self.loss,model.vars()))#objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        #self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (x0,u, ts), true_x = minibatch
        pred_x = self.rollout(x0,u,ts[0])
        return jnp.mean((pred_x - true_x)**2)

    def metrics(self, loader):
        mse = lambda mb: np.asarray(self.loss(mb))
        return {"MSE": self.evalAverageMetrics(loader, mse)}
    
    def logStuff(self, step, minibatch=None):
        # loader = self.dataloaders['test']
        # metrics = {'test_Rollout': np.exp(self.evalAverageMetrics(loader,partial(log_rollout_error,loader.dataset,self.model)))}
        # self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)

from mujoco_models import DeltaNN
import mujoco_models
from datasets import MujocoRegression

def makeTrainer(*,dataset=MujocoRegression,network=DeltaNN,num_epochs=2000,ndata=20000,seed=2021,aug=False,
                bs=500,lr=3e-3,device='cuda',split={'train':3000,'test':1000},
                net_config={'num_layers':3,'ch':128},log_level='warn',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},},#'early_stop_metric':'val_MSE'},
                save=False,):

    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata,chunk_len=5)
        datasets = split_dataset(base_ds,splits=split)
    model = network(base_ds.xdim,base_ds.udim,**net_config)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    #equivariance_test(model,dataloaders['train'],net_config['group'])
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    return RolloutTrainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    cfg = argupdated_config(makeTrainer.__kwdefaults__,namespace=(mujoco_models,))
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])
    # Trial = ode_trial(makeTrainer)
    # cfg,outcome = Trial()
    # print(outcome)
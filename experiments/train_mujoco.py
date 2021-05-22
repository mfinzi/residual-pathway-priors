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
import pandas as pd
import jax

levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                    'warn': logging.WARNING,'warning': logging.WARNING,
                    'info': logging.INFO,'debug': logging.DEBUG}

from trainer.classifier import Regressor,Classifier

def _rollout(model,x0,u,ts):
    return model.rollout(x0,u,ts)

def rollout(model):
    return objax.Jit(vmap(partial(_rollout,model),(0,0,None)),model.vars())


def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

def log_rollout_error(roller,k,minibatch):
    (z0, u,ts), gt_zts = minibatch
    pred_zt,_ = roller(z0,u[:,:k],ts[0][:k])
    errs = vmap(vmap(rel_err))(pred_zt,gt_zts[:,:k]) # (bs,T,)
    clamped_errs = jax.lax.clamp(1e-7,errs,np.inf)
    log_geo_mean = jnp.log(clamped_errs).mean()
    return log_geo_mean

class RolloutTrainer(Regressor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.rollout = rollout(model)
        self.loss = objax.Jit(self.loss,model.vars())
        #self.model = objax.Jit(self.model)
        self.gradvals = objax.Jit(objax.GradValues(self.loss,model.vars()))#objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        #self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
    
    # def mse(self,minibatch):
    #     (x0,u, ts), true_x = minibatch
    #     pred_x,reg = self.rollout(x0,u,ts[0])
    #     return jnp.mean((pred_x - true_x)**2)

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (x0,u, ts), true_x = minibatch
        pred_x,reg = self.rollout(x0,u,ts[0])
        # for k, v in self.model.vars().items():
        #     print(k,'NODE' in k)
        #l2 = sum((v.value ** 2).sum() for k, v in self.model.model.vars().items() if k.endswith('w'))
        return jnp.mean((pred_x - true_x)**2)+.01*jnp.mean(reg)

    def metrics(self, loader):
        #l2 = sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('w'))
        def _metrics(minibatch):
            (x0,u, ts), true_x = minibatch
            pred_x,reg = self.rollout(x0,u,ts[0])
            mse = np.asarray(jnp.mean((pred_x - true_x)**2))
            reg = np.asarray(jnp.mean(reg))
            return pd.Series({'MSE':mse,'reg':reg})
        return self.evalAverageMetrics(loader,_metrics)
        #mse = lambda mb: np.asarray(self.mse(mb))
        #return {"MSE": self.evalAverageMetrics(loader, mse)}
    
    def logStuff(self, step, minibatch=None):
        extra_metrics = {}
        loader = self.dataloaders['_test_episodes']
        for k in [10,30,100]:
            extra_metrics[f'rgre@{k}']=np.exp(self.evalAverageMetrics(loader,partial(log_rollout_error,self.rollout,k)))
        self.logger.add_scalars('metrics', extra_metrics, step)
        super().logStuff(step,minibatch)

from mujoco_models import DeltaNN
import mujoco_models
from datasets import MujocoRegression,MujocoRollouts

def makeTrainer(*,network=DeltaNN,num_epochs=2000,ndata=50000,seed=2021,aug=False,
                bs=500,lr=1e-3,device='cuda',split={'train':6000,'test':1000},env='HopperFull-v0',
                net_config={'num_layers':2,'ch':128},log_level='warn',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.05,'timeFrac':.1},},#'early_stop_metric':'val_MSE'},
                save=False,):

    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = MujocoRegression(N=ndata,env=env)
        datasets = split_dataset(base_ds,splits=split)
        datasets['_test_episodes'] = MujocoRollouts(N=100,env=env)
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
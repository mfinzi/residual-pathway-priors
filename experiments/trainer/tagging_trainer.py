import torch
import torch.nn as nn
from oil.utils.utils import export
from .classifier import Classifier
import jax
import jax.numpy as jnp
import numpy as np
import objax
from sklearn.metrics import roc_auc_score

@export
class TaggingTrainer(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """


    def metrics(self,loader):
        y_probs,y_trues = zip(*[(jax.nn.softmax(self.model.predict(mb[0]),-1)[:,1],mb[1]) for mb in loader])
        y_probs,y_trues = np.concatenate(y_probs),np.concatenate(y_trues)
        auroc = roc_auc_score(y_trues,y_probs)
        acc = ((y_probs>0)==y_trues).mean()
        return {'Acc':acc,'AUROC':auroc}
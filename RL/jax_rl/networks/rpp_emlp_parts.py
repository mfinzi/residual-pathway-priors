#%%
from emlp.reps import Rep
from emlp.nn import uniform_rep
from rpp.flax import MixedEMLPBlock,MixedLinear,Sequential,EMLPBlock
from oil.utils.utils import Named,export
import logging

def parse_rep(ch,group,num_layers):
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    return middle_layers

@export
def HeadlessRPPEMLP(rep_in,group,ch=384,num_layers=3):
    logging.info("Initing RPP-EMLP (flax)")
    rep_in = rep_in(group)
    reps = [rep_in]+parse_rep(ch,group,num_layers)
    logging.info(f"Reps: {reps}")
    return Sequential(*[MixedEMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])])

def HeadlessEMLP(rep_in,group,ch=384,num_layers=3):
    logging.info("Initing RPP-EMLP (flax)")
    rep_in = rep_in(group)
    reps = [rep_in]+parse_rep(ch,group,num_layers)
    logging.info(f"Reps: {reps}")
    return Sequential(*[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])])

# %%

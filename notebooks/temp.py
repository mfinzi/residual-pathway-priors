import jax
import emlp
from emlp import EMLP, EMLPH, MLPH
from emlp.reps import Scalar, T
import sys
sys.path.append("../trainer/")
from hamiltonian_dynamics import DoubleSpringPendulum, BHamiltonianFlow, HamiltonianDataset, WindyDoubleSpringPendulum
import jax.numpy as jnp
sys.path.append("..")
from soft_emlp import MixedEMLP
from emlp.groups import SO,O,S,Z,Trivial
from emlp.groups import SO2eR3,O2eR3,DkeR3
import numpy as np

from torch.utils.data import DataLoader
from utils import LoaderTo
from oil.datasetup.datasets import split_dataset


from IPython.display import HTML
ds = WindyDoubleSpringPendulum(n_systems=100, chunk_len=50)
HTML(ds.animate())



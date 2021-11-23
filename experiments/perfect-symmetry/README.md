# Mis-Specified Symmetry Experiments

These scripts contain the code to reproduce the experiments in which the data exhibit perfect symmetry, but the symmetry in the model is incorrectly specified, as in Figure 2a in the paper.

To produce the results on the Inertia task run:
```{bash}
python inertia_runner.py --network={emlp, mixedemlp, mlp}
```


To produce the results on the Pendulum task run:
```{bash}
python pendulum_runner.py --network={emlp, mixedemlp, mlp}
```

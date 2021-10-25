# Approximate Symmetry Experiments


### Ablations
To train a suite of models with different priors on the windy pendulum data you just need to run:
```{bash}
source windy_runner.sh
```

To train a suite of models with different priors on the modified inertia dataset you just need to run:
```{bash}
source inertia_runner.sh
```

With these ablations run you should be able to use the `plotter` notebook to reproduce Figure 2b in the paper.
# Learning Levels of Equivariance

This directory contains the code to reproduce Figure 3 (Left). First you'll need to train a set of RPP models on both the Inertia and Modified Inertia data which can be accomplisheed with the following:
```{bash}
python modified_inertia.py --modified={T, F}
```

Then you should have what you need to produce the plot using the notebook.
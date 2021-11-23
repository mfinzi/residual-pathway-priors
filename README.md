# residual-pathway-priors

# Installation instructions
To run the scripts you will instead need to clone the repo and install it locally which you can do with
```bash
git clone https://github.com/mfinzi/residual-pathway-priors.git
cd residual-pathway-priors
pip install -e .
```

# Experimental results
- To reproduce the reinforcment learning results from the paper, see the `RL/` directory.
- To reproduce the results in cases with exact symmetries (Figure 2a), see `experiments/perfect-symmetry/`
- To reproduce the results in cases with approximate symmetries (Figures 2b & 7), see `experiments/prior-var-ablation/`
- To reproduce the results in cases with mis-specified symmetries (Figure 2c), see `experiments/misspec-symmetry/`
- To reproduce the UCI results in Table 1, see `experiments/UCI/`
- To reproduce the CIFAR-10 resultsin Table 1, see `experiments/cifar/`


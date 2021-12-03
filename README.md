# Residual Pathway Priors for Soft Equivariance Constraints
This repo contains the implementation and the experiments for the paper 

[Residual Pathway Priors for Soft Equivariance Constraints](https://arxiv.org/abs/2112.01388)


<img src="https://user-images.githubusercontent.com/12687085/144558785-f86f0da1-9176-4ff1-8047-bd9167022153.jpeg" width=450> <img src="https://user-images.githubusercontent.com/12687085/144559340-95d7739d-f368-4861-9f78-3baa0de0ab75.png" width=350>


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

If you find our work helpful, cite it with
```bibtex
@article{finzi2021residual,
  title={Residual Pathway Priors for Soft Equivariance Constraints},
  author={Finzi, Marc and Benton, Gregory and Wilson, Andrew G},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

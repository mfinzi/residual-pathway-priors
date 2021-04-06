# residual-pathway-priors

# Installation instructions
To run the scripts you will instead need to clone the repo and install it locally which you can do with
```bash
git clone https://github.com/mfinzi/residual-pathway-priors.git
cd residual-pathway-priors
pip install -e .
```

# Experimental results
To train the standard MLP on the modified inertia task, you can run
```
python experiments/train_regression.py --network MLP --num_epochs 1000
```
Likewise for the standard EMLP

```
python experiments/train_regression.py --network EMLP --num_epochs 1000
```
And the residual pathway prior EMLP
```
python train_regression.py --network MixedEMLP --num_epochs 1000
```

These should produce the following results:

|| Train MSE | Test MSE  | O(3) Equiv error  |
|---------|------|------|------|
|MLP |   0.034  | 4.78 | 0.17 |
|EMLP|   0.235 | 2.39 | 2e-7 |
|MixedEMLP|   0.070 | **0.15** | 0.08 |

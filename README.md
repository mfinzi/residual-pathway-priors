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
python experiments/train_regression.py --network MixedEMLP --num_epochs 1000
```

These should produce the following results:

|| Train MSE | Test MSE  | O(3) Equiv error  |
|---------|------|------|------|
|MLP |   0.034  | 4.78 | 0.17 |
|EMLP|   0.235 | 2.39 | 2e-7 |
|MixedEMLP|   0.070 | **0.15** | 0.08 |

New numbers from recent run with improved branch RPP-EMLP and wd_basic=1.0:
|MixedEMLP|   0.081 | **0.21** | 0.09 |

To train the models on the windy pendulum ...

<div align="center">
<img src="https://user-images.githubusercontent.com/12687085/115454965-5c233900-a1ef-11eb-9f83-6d94e1edf3d1.gif" width="500"/>
</div>

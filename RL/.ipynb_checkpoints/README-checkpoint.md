# RPP-RL

This repo is a fork of the Jax-RL repo (https://github.com/ikostrikov/jax-rl) and contains the code to train RPP-EMLPs on Mujoco environments.  

# Run

```
RPP Policy only
```bash
python train_rpp.py --env_name=Ant-v2 --save_dir=./tmp/rpp --rpp_value=False
```
For the policy function you can control weight decay of the basic and equivariant layers with the `actor_basic_wd` and `actor_equiv_wd` arguments respectively. 

RPP Policy and Value function
```bash
python train_rpp.py --env_name=Ant-v2 --save_dir=./tmp/rppv --rpp_value=True
```


You can swap out the group with the `--group` argument. For exmaple:
```bash
python train_rpp.py --env_name=Ant-v2 --save_dir=./tmp/rpp_O2 --rpp_value=False --group="O(2)"
```
to train with the O(2) symmetry group instead of SO(2).
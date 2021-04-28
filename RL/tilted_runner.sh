#!/bin/bash

for basic_wd in {1,1e-2,1e-4,1e-6,0}
do
    for equiv_wd in {1,1e-2,1e-4,1e-6,0}
    do
        python tilted_cartpole_runner.py --equiv_wd=${equiv_wd} --basic_wd=${basic_wd} --network=MixedEMLP
    done
done
echo All done
#!/bin/bash

for basic_wd in {1e-1,1e-2,1e-3,1e-4,1e-5,0}
do
    for equiv_wd in {1e-1,1e-2,1e-3,1e-4,1e-5,0}
    do
        python windy_pendulum_runner.py --equiv_wd=${equiv_wd} --basic_wd=${basic_wd} > ./logs/basic${basic_wd}_${equiv_wd}.txt
    done
done
echo All done
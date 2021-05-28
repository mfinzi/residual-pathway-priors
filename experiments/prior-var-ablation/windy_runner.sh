#!/bin/bash

for basic_wd in {1e-3,1e-2,1e-1,1.0,10.,100.}
do
    for equiv_wd in {1e-2,1e-3,1e-4,1e-5}
    do
        python windy_pendulum_runner.py --equiv_wd=${equiv_wd} --basic_wd=${basic_wd}
    done
done
echo All done
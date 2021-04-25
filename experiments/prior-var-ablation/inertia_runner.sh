#!/bin/bash

for basic_wd in {1e-2,1e-3,1e-4,1e-5,1e-6}
do
    for equiv_wd in {1e-2,1e-3,1e-4,1e-5,1e-6}
    do
        python modified_inertia.py --equiv_wd=${equiv_wd} --basic_wd=${basic_wd}
    done
done
echo All done
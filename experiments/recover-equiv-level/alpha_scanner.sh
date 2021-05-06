#!/bin/bash
for trial in {0..10}
do
    for alpha in $(seq 0.0 0.1 1.0)
    do
            echo $trial
            python prior_interpolation.py --trial=${trial} --alpha=${alpha}

    done
done
echo All done
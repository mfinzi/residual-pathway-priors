#!/bin/bash
for trial in {0..10}
do
    for wind_scale in $(seq 0.012 0.002 0.04)
    do
            echo $basic_wd $trial
            python windy_pendulum_runner.py --trial=${trial} --wind_scale=${wind_scale}

    done
done
echo All done
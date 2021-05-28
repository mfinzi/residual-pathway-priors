#!/bin/bash
for wind_level in 1e-4 1e-2 5e-2 1e-1 5e-1
do
    echo $wind_level
    python windy_pendulum_runner.py --wind_level=$wind_level --basic_wd=1e-3
done
echo All done
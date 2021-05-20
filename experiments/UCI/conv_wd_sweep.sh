#!/bin/bash
for trial in {1e-6,1e-4,1e-2,1.0}
do
            python train_uci.py --dataset=$1 --conv_wd=$conv_wd --basic_wd=1e-6
done
echo All done
#!/bin/bash
declare -a StringArray=('energy' 'fertility' 'pendulum' 'wine')

for dataset in "${StringArray[@]}";
do
    for trial in {0..9}
    do
        python train_uci.py --dataset=${dataset} --conv_wd=1e-4 --basic_wd=0.01 --trial=$trial --network=$1
    done
done
echo All done
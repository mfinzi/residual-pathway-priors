#!/bin/bash

declare -a StringArray=('yacht' 'airfoil' 'bike' 'breastcancer' 'buzz' 'concrete' 'elevators' 'energy' 'fertility' 
                    'forest' 'gas' 'housing' 'keggdirected' 'keggundirected' 'pendulum' 'protein' 'skillcraft' 'wine')

for dataset in "${StringArray[@]}";
do
    echo $dataset
    python train_uci.py --network=$1 --conv_wd=1e-4 --basic_wd=.1 --dataset=$dataset
done
echo All done
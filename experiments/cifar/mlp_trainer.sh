#!/bin/bash
for trial in {0..10}
do
    echo $trial
    python cifar_trainer.py --network=mlp --trial=$trial --basic_wd=1e-4 --conv_wd=0.0 > ./logs/mlp${trial}_basicwd0.0001.txt
done
echo All done
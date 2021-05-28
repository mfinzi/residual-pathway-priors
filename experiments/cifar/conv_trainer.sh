#!/bin/bash
for trial in {0..10}
do
    echo $trial
    python cifar_trainer.py --network=conv --trial=$trial --basic_wd=0. --conv_wd=1e-4  > ./logs/conv${trial}_convwd0.0001.txt
done
echo All done
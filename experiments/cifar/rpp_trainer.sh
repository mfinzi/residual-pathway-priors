#!/bin/bash
for trial in {0..10}
do
    echo $trial
    python cifar_trainer.py --trial=$trial --basic_wd=$1 --conv_wd=1e-4  > ./logs/model${trial}_$1.txt
done
echo All done
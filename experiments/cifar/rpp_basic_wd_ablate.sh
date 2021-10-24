#!/bin/bash
for basic_wd in {1e-4,1e-3,1e-2,1e-1}
do
    python cifar_trainer.py --trial=0 --basic_wd=$basic_wd --conv_wd=1e-3  > ./logs/model0_basic${basic_wd}_conv$1.txt
done
echo All done
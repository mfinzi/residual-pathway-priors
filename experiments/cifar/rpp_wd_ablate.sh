#!/bin/bash
for conv_wd in {1e-4,1e-2}
do
    python cifar_trainer.py --trial=0 --basic_wd=$1 --conv_wd=$conv_wd  > ./logs/model0_basic$1_conv${conv_wd}.txt
done
echo All done
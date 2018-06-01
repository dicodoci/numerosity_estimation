#!/bin/bash
python train.py --z_dim 20 --num_epochs 100 > output1.txt
python train2.py --z_dim 20 --num_epochs 100 --model_name "234" > output2.txt



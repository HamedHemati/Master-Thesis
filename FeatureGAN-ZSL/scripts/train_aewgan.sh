#!/bin/bash

data_path='/data/cvg/hamed/Datasets/xlsa17/data'
outputs_path='/data/cvg/hamed/Outputs'
                               
cd ..


python main.py --cuda --batch-size 64 --n-epochs 50 \
                --data-path $data_path --outputs-path $outputs_path \
                --model 'AEWGAN' --use-cent-loss 'no' --lambda-centl 1.0\
                --dataset 'AWA2' --use-valset 'no' --eval-zsl 'yes' --eval-gzsl 'yes'\
                --feat-type 'res101' --cls-emb-type 'att'\
                --n-iter-d 5 --lr 0.00001 --lambda-gp 10.0 --cls-weight 0.01 --n-cls-epochs 30\
                --n-z 100 --n-h-d 4096 --n-h-g 4096 \
                --n-synth-samples 2500 --save-every 0 \
                --name-index '30' --random-seed 67232


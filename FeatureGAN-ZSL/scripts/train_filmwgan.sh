#!/bin/bash

data_path='/data/cvg/hamed/Datasets/xlsa17/data'
outputs_path='/data/cvg/hamed/Outputs'

cd ..              


python main.py --cuda --batch-size 64 --n-epochs 50 \
                --data-path $data_path --outputs-path $outputs_path \
                --model 'FiLMWGAN' --use-cent-loss 'no' --lambda-centl 1.0 \
                --dataset 'AWA1' --use-valset 'no' --eval-zsl 'yes' --eval-gzsl 'yes' \
                --feat-type 'res101' --cls-emb-type 'att' \
                --n-iter-d 5 --lr 0.00001 --lambda-gp 10.0 --cls-weight 0.01 --n-cls-epochs 25 \
                --n-z 100 --n-h-d 4400 --n-h-g 4400 \
                --n-synth-samples 2500 --save-every 0 \
                --name-index '6' --random-seed 67232

python main.py --cuda --batch-size 64 --n-epochs 50 \
                --data-path $data_path --outputs-path $outputs_path \
                --model 'FiLMWGAN' --use-cent-loss 'no' --lambda-centl 1.0 \
                --dataset 'AWA2' --use-valset 'no' --eval-zsl 'yes' --eval-gzsl 'yes' \
                --feat-type 'res101' --cls-emb-type 'att' \
                --n-iter-d 5 --lr 0.00001 --lambda-gp 10.0 --cls-weight 0.01 --n-cls-epochs 25 \
                --n-z 100 --n-h-d 4400 --n-h-g 4400 \
                --n-synth-samples 2500 --save-every 0 \
                --name-index '7' --random-seed 67232 

#python main.py --cuda --batch-size 64 --n-epochs 100 \
#                --data-path $data_path --outputs-path $outputs_path \
#                --model 'FiLMWGAN' --use-cent-loss 'no' --lambda-centl 1.0 \
#                --dataset 'CUB' --use-valset 'no' --eval-zsl 'yes' --eval-gzsl 'yes' \
#                --feat-type 'res101' --cls-emb-type 'att' \
#                --n-iter-d 5 --lr 0.00001 --lambda-gp 10.0 --cls-weight 0.01 --n-cls-epochs 25 \
#                --n-z 300 --n-h-d 4400 --n-h-g 4400 \
#                --n-synth-samples 500 --save-every 0 \
#                --name-index '8' --random-seed 67232 

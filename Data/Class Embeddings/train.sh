#!/bin/bash

fastText_model_path='/data/cvg/hamed/Pre-trained_Models/fastText/wiki.en/wiki.en.bin'


python train.py --dataset "cub" --model "lstm" \
				--hidden-size 250 --n-epochs 70 --lr 0.005 \
				--fastText-model-path $fastText_model_path

#!/bin/bash

fastText_model_path='/data/cvg/hamed/Pre-trained_Models/fastText/wiki.en/wiki.en.bin'
checkpoint_path='outputs/awa2_gru_hidden-size150_nitr20_lr0.005/checkpoint.pth'

python test.py --dataset "awa2" --model "gru" \
				--hidden-size 150 --checkpoint-path $checkpoint_path --method "rnn-mean" \
				--hdf "no" --tsne "yes" \
				--fastText-model-path $fastText_model_path


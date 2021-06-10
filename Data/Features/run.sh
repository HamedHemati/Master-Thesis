#!/usr/bin/bash

python feature_extractor.py --ds-path '/data/cvg/hamed/Datasets/CUB' \
               --ds-name 'CUB' \
               --xlsa17-path '/data/cvg/hamed/Datasets/xlsa17/data'\
               --save-path '/data/cvg/hamed/Datasets/CUB-ZSL/alexnet' \
               --model 'alexnet' \
               --model-checkpoint '/data/cvg/hamed/Pre-trained_Models/alexnet/alexnet-owt-4df8aa71.pth' \
               --num-workers 1 \
               --batch-size 8

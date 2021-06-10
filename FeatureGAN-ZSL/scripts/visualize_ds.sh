#!/bin/bash
cd ..
cd utils

python visualize_ds_features.py --dataset 'AWA2'  --feat-type 'res18' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 

python visualize_ds_features.py --dataset 'AWA2'  --feat-type 'vgg16-bn' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 

python visualize_ds_features.py --dataset 'AWA2'  --feat-type 'vgg19-bn' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 

python visualize_ds_features.py --dataset 'CUB'  --feat-type 'res18' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 

python visualize_ds_features.py --dataset 'CUB'  --feat-type 'vgg16-bn' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 

python visualize_ds_features.py --dataset 'CUB'  --feat-type 'vgg19-bn' --n-samples 50 \
				--draw-trainset 'yes' --draw-testset 'yes' \
                --data-path '/data/cvg/hamed/Datasets/xlsa17/data' \
                --output-path '/data/cvg/hamed/Outputs/DS_Visualization' 


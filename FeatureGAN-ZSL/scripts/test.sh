#!/bin/bash

data_path='/data/cvg/hamed/Datasets/xlsa17/data'
images_path='/data/cvg/hamed/Datasets'
outputs_path='/data/cvg/hamed/Outputs'
                               
cd ..



python test.py  --name-index 5 --classifier-type 'nc' --n-synth-samples 50\
                --compute-confusion "" --tsne-synth 'yes' --zsc "no" --gzsc "no" --zsr "no" --gzsr "no" \
                --data-path $data_path --outputs-path $outputs_path --images-path $images_path

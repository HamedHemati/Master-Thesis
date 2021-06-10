# Master Thesis: Zero-Shot Learning using Generative Adversarial Networks
My master thesis on zero-shot learning that I did in the Computer Vision Group at the University of Bern

[You can download the thesis paper from here](https://drive.google.com/file/d/1n9c-BGRVZ8-Be7sY3TmqPss9MVPELIxM/view?usp=sharing)

## FeatureGAN-ZSL

The folder "Feature GAN - ZSL" contains the codes required to train and test feature generative adversarial networks. All train and test scripts are inside "scripts" folder which can be simply executed using bash command.

1) to run train: first set the `data_path` and `outputs_path` variables which are the paths to datasets and output folders (logs) repectively.

1) to run test:  first set the `data_path`, `image_path` and `outputs_path` variables which are the paths to datasets, folders of datasets which contain the images and outputs path. Then you need to set the index number of the output fodler that you want to execute test on and run the shell by bash command.

* the structure of folders for datasets are described in "Data" folder of this DVD.



## Data

The folder "ZSL Dataset Tools contains two folder `Class Embeddings` and `Features`:

1) Class Embeddings: contains all source code and data required to extract embedding vector from textual descriptions. The run the train script for fastText mode, you need to set the path to the fastText model (wiki.en.bin). The fastText pretrained models can be downloaded from the link below:

https://fasttext.cc/docs/en/crawl-vectors.html

To extract embedding vectors, you need run the test.sh script by setting the path of the checkpoint created during training. The outputs will be saves in the folder of the checkpoint.

2) Features : contains codes and scripts for extracting features from pre-trained CNN models. 

To extract features you need enter path of the datasets and the pre-trained model (of PyTorch) which can be downloaded directly (from its source code on Github). 
ex) 
`alexnet: https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth`

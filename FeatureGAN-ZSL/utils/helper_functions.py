import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from os import listdir
from os.path import join 


# ---------- Neural Network
def weights_init(m):
    """Initializes weights of a neural network
    """
    for p in m.parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)


def weights_init_2(m):
    """Initializes weights of a neural network
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)  


# ---------- Data Loader and Initializers
def init_env(random_seed):
    """Sets random seeds for better random generation
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    #cudnn.benchmark = True


def get_new_labels(labels, classes):
    """Return new labels with respect to the given classes
    """
    new_labels = torch.LongTensor(labels.size())
    for i in range(classes.size(0)):
        new_labels[labels==classes[i]] = i 
    return new_labels


def get_min_max_epoch(path):
    """Returns max epoch index in a directory of checkpoints
    """
    files = listdir(path)
    indices = [int(f.split('_')[-1].split('.')[0]) for f in files]
    return min(indices), max(indices)


def read_param_file(opt):
    """Reads arguments from file.
    """
    folder_name = [d for d in listdir(opt.outputs_path) if d.startswith(str(opt.name_index) + '_')][0]
    opt.path = join(opt.outputs_path, folder_name)
    print('output path:\n' + opt.path)

    with open(join(opt.path, 'params.txt'), 'r') as params_file:
        lines = params_file.readlines()
    lines = [p.strip().split(':') for p in lines]
    params = {}
    for p in lines: 
        params.update({p[0]: p[1]})
    opt.model = params['model']
    opt.n_z = int(params['n_z'])
    opt.n_feat = int(params['n_feat'])
    opt.n_cls_emb = int(params['n_cls_emb'])
    opt.n_h_g = int(params['n_h_g'])
    opt.dataset = params['dataset']
    opt.feat_type = params['feat_type']
    opt.cls_emb_type = params['cls_emb_type']
    return opt
    
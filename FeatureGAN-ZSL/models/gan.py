import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.helper_functions import weights_init


# ==================== GAN
class NetD_GAN(nn.Module):
    def __init__(self, opt): 
        super(NetD_GAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat + opt.n_cls_emb, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x, emb):
        x = torch.cat((x, emb), 1) 
        x = self.lrelu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class NetG_GAN(nn.Module):
    def __init__(self, opt):
        super(NetG_GAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_cls_emb + opt.n_z, opt.n_h_g)
        self.fc2 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)

    def forward(self, z, emb):
        x = torch.cat((z, emb), 1)
        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.helper_functions import weights_init


# ==================== ZSWGAN
class NetD_ZSWGAN(nn.Module):
    def __init__(self, opt): 
        super(NetD_ZSWGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat + opt.n_cls_emb, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        x = torch.cat([x, emb], 1) 
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetG_ZSWGAN(nn.Module):
    def __init__(self, opt):
        super(NetG_ZSWGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_cls_emb + opt.n_z, opt.n_h_g)
        self.fc2 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)

    def forward(self, z, emb):
        x = torch.cat([z, emb], 1)
        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x   


# ==================== ZSWGAN2
class NetD_ZSWGAN2(nn.Module):
    def __init__(self, opt): 
        super(NetD_ZSWGAN2, self).__init__()
        self.fc_emb = nn.Linear(opt.n_cls_emb, opt.n_z)
        self.fc1 = nn.Linear(opt.n_feat + opt.n_z, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        emb = self.lrelu(self.fc_emb(emb))
        x = torch.cat([x, emb], 1) 
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetG_ZSWGAN2(nn.Module):
    def __init__(self, opt):
        super(NetG_ZSWGAN2, self).__init__()
        self.fc_emb = nn.Linear(opt.n_cls_emb, opt.n_z)
        self.fc1 = nn.Linear(2 * opt.n_z, int(opt.n_h_g))
        self.fc2 = nn.Linear(int(opt.n_h_g), opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self, z, emb):
        emb = self.tanh(self.fc_emb(emb))
        x = torch.cat([emb, z], 1)
        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.helper_functions import weights_init


# ==================== fCLS-WGAN
class NetD_WGAN(nn.Module):
    def __init__(self, opt): 
        super(NetD_WGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat + opt.n_cls_emb, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        x = torch.cat([x, emb], dim=1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetG_WGAN(nn.Module):
    def __init__(self, opt):
        super(NetG_WGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_z + opt.n_cls_emb, opt.n_h_g)
        self.fc2 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)

    def forward(self, z, emb):
        x = torch.cat([z, emb], dim=1)
        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
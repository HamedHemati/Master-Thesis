import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.helper_functions import weights_init


# ==================== FiLM Module
class FiLMMod(nn.Module):
    def __init__(self, n_cls_emb, n_out):
        super(FiLMMod, self).__init__()
        self.fc_gamma = nn.Linear(n_cls_emb, n_out)
        self.fc_beta = nn.Linear(n_cls_emb, n_out)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.apply(weights_init)
        
    def forward(self, x, cls_emb, use_actv=False):
        gamma = self.fc_gamma(cls_emb)
        beta = self.fc_beta(cls_emb)
        if use_actv:
            gamma = self.lrelu(gamma)
            beta = self.lrelu(beta)
        return (gamma * x) + beta


# ==================== FiLMWGAN
class NetD_FiLMWGAN(nn.Module):
    def __init__(self, opt): 
        super(NetD_FiLMWGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, int(opt.n_h_d/2))
        self.film = FiLMMod(opt.n_cls_emb, int(opt.n_h_d/2))
        self.fc3 = nn.Linear(int(opt.n_h_d/2), int(opt.n_h_d/4))
        self.fc4 = nn.Linear(int(opt.n_h_d/4), 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.film(x, emb, use_actv=True)
        x = self.lrelu(self.fc3(x))
        x = self.fc4(x)
        return x


class NetG_FiLMWGAN(nn.Module):
    def __init__(self, opt):
        super(NetG_FiLMWGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_z, int(opt.n_h_g/4))
        self.film = FiLMMod(opt.n_cls_emb, int(opt.n_h_g/4))
        self.fc2 = nn.Linear(int(opt.n_h_g/4), opt.n_h_g)
        self.fc3 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, z, emb):
        x = self.lrelu(self.fc1(z))
        x = self.film(x, emb, use_actv=True)
        x = self.lrelu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# ==================== FiLMWGAN2
class NetD_FiLMWGAN2(nn.Module):
    def __init__(self, opt): 
        super(NetD_FiLMWGAN2, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat, opt.n_h_d)
        self.fc2 = nn.Linear(int(opt.n_h_d), int(opt.n_h_d/16))
        self.film = FiLMMod(opt.n_cls_emb, int(opt.n_h_d/16))
        self.fc3 = nn.Linear(int(opt.n_h_d/16), int(opt.n_h_d/32))
        self.fc4 = nn.Linear(int(opt.n_h_d/32), 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.film(x, emb, use_actv=False)
        x = self.lrelu(self.fc3(x))
        x = self.fc4(x)
        return x


class NetG_FiLMWGAN2(nn.Module):
    def __init__(self, opt):
        super(NetG_FiLMWGAN2, self).__init__()
        self.fc1 = nn.Linear(opt.n_z, int(opt.n_h_g/16))
        self.film = FiLMMod(opt.n_cls_emb, int(opt.n_h_g/16))
        self.fc2 = nn.Linear(int(opt.n_h_g/16), opt.n_h_g)
        self.fc3 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, z, emb):
        x = self.lrelu(self.fc1(z))
        x = self.film(x, emb, use_actv=False)
        x = self.lrelu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

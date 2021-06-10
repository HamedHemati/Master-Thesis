import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.helper_functions import weights_init


# ==================== AEWGAN
class NetD_AEWGAN(nn.Module):
    def __init__(self, opt): 
        super(NetD_AEWGAN, self).__init__()
        self.fc1 = nn.Linear(opt.n_feat + opt.n_cls_emb, opt.n_h_d)
        self.fc2 = nn.Linear(opt.n_h_d, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(weights_init)

    def forward(self, x, emb):
        x = torch.cat([x, emb], 1) 
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetG_AEWGAN(nn.Module):
    def __init__(self, opt):
        super(NetG_AEWGAN, self).__init__()
        self.fc_emb1 = nn.Linear(opt.n_cls_emb, int(opt.n_cls_emb/2))
        #self.fc_emb2 = nn.Linear(int(opt.n_cls_emb/2), int(opt.n_cls_emb/4))
        #self.fc_emb_inv_1 = nn.Linear(int(opt.n_cls_emb/4), int(opt.n_cls_emb/2))
        self.fc_emb_inv_2 = nn.Linear(int(opt.n_cls_emb/2), opt.n_cls_emb)

        self.fc1 = nn.Linear(opt.n_cls_emb + opt.n_z, opt.n_h_g)
        self.fc2 = nn.Linear(opt.n_h_g, opt.n_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)

    def forward(self, z, emb):
        emb = self.lrelu(self.fc_emb1(emb))
        #emb = self.lrelu(self.fc_emb2(emb))
        #emb_reconstr = self.lrelu(self.fc_emb_inv_1(emb))
        emb_reconstr = self.relu(self.fc_emb_inv_2(emb))
        x = torch.cat([z, emb], 1)
        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x, emb_reconstr

import torch
import torch.nn as nn
from .helper_functions import weights_init


class LinearSM(nn.Module):
    def __init__(self, n_feat, n_cls):
        super(LinearSM, self).__init__()
        self.fc = nn.Linear(n_feat, n_cls)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.log_softmax(self.fc(x))
        return x
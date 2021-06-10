import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .feature_ds import fds_dataloader
from .lin_sm import LinearSM


class FeatSoftmaxClassifier(object):
    def __init__(self, X, Y, n_feat, n_cls, n_epochs):
        self.cuda = torch.cuda.is_available()
        self.num_epochs = n_epochs
        self.X = X
        self.Y = Y
        self.model = LinearSM(n_feat, n_cls)
        self.criterion = nn.NLLLoss()
        if self.cuda:
            self.model, self.criterion = self.model.cuda(), self.criterion.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.ds_loader = fds_dataloader(X=X, Y=Y, batch_size=65, shuffle=True)

    def fit(self):
        print("Training the softmax classifier:")
        self.model.train()
        for epoch in range(self.num_epochs):
            loss_sum = 0.0
            for i,(x,y) in enumerate(self.ds_loader):
                self.model.zero_grad()
                if self.cuda:
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
            print("Epoch %d, loss = %.4f" % (epoch, loss_sum/len(self.ds_loader)))
        print("\n\n")    
        self.model.eval()

    def predict(self, X):  
        if self.cuda:
            X = X.cuda()
        return self.model(X)
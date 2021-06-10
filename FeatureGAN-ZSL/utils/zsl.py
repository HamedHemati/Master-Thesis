import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from .helper_functions import weights_init, get_new_labels
from sklearn.preprocessing import MinMaxScaler 
import sys
from .feature_ds import fds_dataloader
from .lin_sm import LinearSM
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm


class ZSL():
    def __init__(self, X, Y, data_handler, batch_size, n_epochs, num_cls, lr, classifier_type="linsoftmax"):
        self.cuda = torch.cuda.is_available()
        self.X = X
        self.Y = Y
        self.data_handler = data_handler
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ds_loader = fds_dataloader(X=X, Y=Y, batch_size=self.batch_size, shuffle=True)
        self.classifier_type = classifier_type
        self._init_classifier(num_cls, lr)
        self.confusion_matrix = None

    def _init_classifier(self, num_cls, lr):
        if self.classifier_type == "linsoftmax":
            self.classifier =  LinearSM(self.X.size(1), num_cls)
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr, betas=(0.5, 0.999))
            if self.cuda:
                self.classifier.cuda()
                self.criterion.cuda()
        elif self.classifier_type == "svm":
            self.classifier = svm.SVC(decision_function_shape='ovo')
        elif self.classifier_type == "nc":
            self.classifier = NearestCentroid()

    def run_zsl(self, compute_confusion=False):
        seen_acc_max = 0.0
        n_cls = self.data_handler.unseen_classes.size(0)
        self.confusion_matrix = np.zeros((n_cls, n_cls))
        if self.classifier_type == "linsoftmax":
            print("Training linsoftmax classifier ...")
            for epoch in range(self.n_epochs):
                loss_avg = 0.0
                for i, (feats,lbls) in enumerate(self.ds_loader):   
                    self.classifier.zero_grad()
                    if self.cuda: 
                        feats, lbls = feats.cuda(), lbls.cuda()  
                    output = self.classifier(feats)
                    loss = self.criterion(output, lbls)
                    loss.backward()
                    self.optimizer.step()
                    loss_avg += loss.item()
                # evaluate zsl for each epoch    
                seen_acc = self._calc_acc_zsl(self.data_handler.test_unseen_features, self.data_handler.test_unseen_labels, self.data_handler.unseen_classes, compute_confusion)
                if seen_acc > seen_acc_max:
                    seen_acc_max = seen_acc
                print("Epoch %d: avg. training loss: %.4f" %(epoch, (loss_avg / float(len(self.ds_loader)))))
        else:
            print("Training " + self.classifier_type + " classifier ...")
            # train the classifier
            self.classifier.fit(self.X.cpu().numpy(), self.Y.cpu().numpy())
            print("Calculating zsl acc ...")
            # evaluate zsl
            seen_acc_max = self._calc_acc_zsl(self.data_handler.test_unseen_features, self.data_handler.test_unseen_labels, self.data_handler.unseen_classes, compute_confusion)
        return (seen_acc_max * 100)

    def _calc_acc_zsl(self, feats, lbls, test_cls, compute_confusion): 
        num_feats = feats.size(0)
        predicted_labels = torch.LongTensor(lbls.size())
        if self.cuda:
            feats = feats.cuda()
        idx_begin = 0
        for i in range(0, num_feats, self.batch_size):
            idx_end = min(num_feats, idx_begin+self.batch_size)
            if self.classifier_type == "linsoftmax":
                output = self.classifier(feats[idx_begin:idx_end]) 
                _, predicted_labels[idx_begin:idx_end] = torch.max(output.data, 1)
            else:
                output = self.classifier.predict(feats[idx_begin:idx_end])  
                predicted_labels[idx_begin:idx_end] = torch.tensor(output).long()
            idx_begin = idx_end
        test_label_final = get_new_labels(lbls, test_cls)
        acc_per_class = torch.FloatTensor(test_cls.size(0)).fill_(0)
        if compute_confusion: # compute the per-class accuracy and confusion matrix
            for i in range(test_cls.size(0)):
                idx = (test_label_final == i)
                self.confusion_matrix[i][2] = i
                for j in range(test_cls.size(0)):
                    self.confusion_matrix[i][j] = np.asscalar(torch.sum(predicted_labels[idx]==j))
                acc_per_class[i] = float(torch.sum(test_label_final[idx]==predicted_labels[idx])) / float(torch.sum(idx))
        else: # only compute the per-class accuracy   
            for i in range(test_cls.size(0)):
                idx = (test_label_final == i)
                acc_per_class[i] = float(torch.sum(test_label_final[idx]==predicted_labels[idx])) / float(torch.sum(idx))
        return acc_per_class.mean()

    def run_gzsl(self, compute_confusion=False):
        acc_seen_max, acc_unseen_max, acc_h_max = 0.0, 0.0, 0.0
        n_all_cls = self.data_handler.seen_classes.size(0) + self.data_handler.unseen_classes.size(0)
        self.confusion_matrix = np.zeros((n_all_cls, n_all_cls))
        if self.classifier_type == "linsoftmax":
            for epoch in range(self.n_epochs):
                loss_avg = 0.0
                for i,  (feats, lbls) in enumerate(self.ds_loader):
                    self.classifier.zero_grad()
                    if self.cuda: 
                        feats, lbls = feats.cuda(), lbls.cuda()
                    output = self.classifier(feats)
                    loss = self.criterion(output, lbls)
                    loss.backward()
                    self.optimizer.step()
                    loss_avg += loss.item()
                acc_seen = self._calc_acc_gzsl(self.data_handler.test_seen_features, self.data_handler.test_seen_labels, self.data_handler.seen_classes, compute_confusion)
                acc_unseen = self._calc_acc_gzsl(self.data_handler.test_unseen_features, self.data_handler.test_unseen_labels, self.data_handler.unseen_classes, compute_confusion)
                acc_h = 2 * (acc_seen * acc_unseen) / (acc_seen + acc_unseen)
                if acc_h > acc_h_max:
                    acc_seen_max, acc_unseen_max, acc_h_max = acc_seen, acc_unseen, acc_h
                print("Epoch %d: avg. training loss: %.4f" %(epoch, (loss_avg/float(len(self.ds_loader)))))          
        else:
            print("Training " + self.classifier_type + " classifier ...")
            self.classifier.fit(self.X.cpu().numpy(), self.Y.cpu().numpy())
            print("Calculating gzsl acc ...")
            acc_seen_max = self._calc_acc_gzsl(self.data_handler.test_seen_features, self.data_handler.test_seen_labels, self.data_handler.seen_classes, compute_confusion)
            acc_unseen_max = self._calc_acc_gzsl(self.data_handler.test_unseen_features, self.data_handler.test_unseen_labels, self.data_handler.unseen_classes, compute_confusion)
            acc_h_max = 2 * (acc_seen_max * acc_unseen_max) / (acc_seen_max + acc_unseen_max)

        return (acc_seen_max * 100), (acc_unseen_max * 100), (acc_h_max * 100)
    
    def _calc_acc_gzsl(self, feats, lbls, test_cls, compute_confusion): 
        num_feats = feats.size(0)
        predicted_labels = torch.LongTensor(lbls.size())
        if self.cuda:
            feats = feats.cuda()
        idx_begin = 0
        for i in range(0, num_feats, self.batch_size):
            idx_end = min(num_feats, idx_begin + self.batch_size)
            if self.classifier_type == "linsoftmax":
                output = self.classifier(feats[idx_begin:idx_end]) 
                _, predicted_labels[idx_begin:idx_end] = torch.max(output.data, 1)
            else:
                output = self.classifier.predict(feats[idx_begin:idx_end])
                predicted_labels[idx_begin:idx_end] = torch.tensor(output).long()
            idx_begin = idx_end
        acc_per_class = 0.0
        if compute_confusion:
            for i in test_cls:
                idx = (lbls == i)
                for j in test_cls:
                    self.confusion_matrix[i][j] = np.asscalar(torch.sum(predicted_labels[idx]==j))
                num_matches = np.asscalar(torch.sum(lbls[idx]==predicted_labels[idx]))
                num_all = np.asscalar(torch.sum(idx))
                acc_per_class += (num_matches / float(num_all))
        else:    
            for i in test_cls:
                idx = (lbls == i)
                num_matches = np.asscalar(torch.sum(lbls[idx]==predicted_labels[idx]))
                num_all = np.asscalar(torch.sum(idx))
                acc_per_class += (num_matches / float(num_all))
        acc_per_class /= float(test_cls.size(0))
        return acc_per_class 
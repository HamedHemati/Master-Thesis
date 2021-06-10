import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as preprocessing
from sklearn import preprocessing
import h5py
from .helper_functions import get_new_labels


class ZSLDataset(Dataset):
    def __init__(self, X, Y, cls_embs, target_classes):
        self.original_Y = Y
        self.cls_embs = cls_embs
        self.X = X
        self.Y = get_new_labels(Y, target_classes)

    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx], self.cls_embs[self.original_Y[idx]]
        
    def __len__(self):
        return len(self.X)
         

class DataHandler(object):
    def __init__(self, opt):
        self._load_image_files(opt)
        if opt.feat_type == 'res101':
            opt.n_feat = 2048
            self._load_files_mat(opt)
        if opt.feat_type == 'res18':
            opt.n_feat = 512
            self._load_files_hdf5(opt)    
        elif opt.feat_type == 'polynet':
            opt.n_feat = 2048
            self._load_files_hdf5(opt)
        elif opt.feat_type == 'vgg19-bn':
            opt.n_feat = 4096
            self._load_files_hdf5(opt)
        elif opt.feat_type == 'vgg16-bn':
            opt.n_feat = 4096
            self._load_files_hdf5(opt)    
        elif opt.feat_type == 'alexnet':
            opt.n_feat = 2048
            self._load_files_hdf5(opt)    
        print("Data loaded successfuly.")

    def _load_image_files(self, opt):
        mat_file = sio.loadmat(opt.data_path + "/" + opt.dataset + "/" + "res101.mat")
        self.image_files = mat_file['image_files'].T.squeeze()

    def _load_files_mat(self, opt):
        file_cls = open(opt.data_path + "/" + opt.dataset + "/" + "allclasses.txt")
        file_cls = file_cls.readlines()
        self.class_names = [c.strip() for c in file_cls]

        mat_file = sio.loadmat(opt.data_path + "/" + opt.dataset + "/" + opt.feat_type + ".mat")
        features = mat_file['features'].T
        opt.n_feat = len(features[0])
        labels = mat_file['labels'].astype(int).squeeze() - 1
        mat_file = sio.loadmat(opt.data_path + "/" + opt.dataset + "/" + "att_splits.mat")
        self.trainval_loc = mat_file['trainval_loc'].squeeze() - 1
        #train_loc = mat_file['train_loc'].squeeze() - 1
        #val_unseen_loc = mat_file['val_loc'].squeeze() - 1
        self.test_seen_loc = mat_file['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = mat_file['test_unseen_loc'].squeeze() - 1
        if opt.cls_emb_type == 'att':
            self.cls_embs = torch.tensor(mat_file['att'].T).float() 
        else: # other types: fastText-context, fastText-names, rnn-lstm, rnn-mean-lstm, tfidf
            cls_emb_file = h5py.File(opt.data_path + "/" + opt.dataset + "/"+ opt.cls_emb_type + '.hdf5', 'r')
            self.cls_embs = torch.tensor(cls_emb_file['cls_embedding'].value).float()
        opt.n_cls_emb = len(self.cls_embs[0])
         
        scaler = preprocessing.MinMaxScaler()    
        scaler.fit(features[self.trainval_loc])
        self.train_features = torch.tensor(scaler.transform(features[self.trainval_loc])).float()
        self.train_features.mul_(1/self.train_features.max())
        self.train_labels = torch.tensor(labels[self.trainval_loc]).long() 
        self.test_seen_features = torch.tensor(scaler.transform(features[self.test_seen_loc])).float() 
        self.test_seen_features.mul_(1/self.train_features.max())
        self.test_seen_labels = torch.tensor(labels[self.test_seen_loc]).long()
        self.test_unseen_features = torch.tensor(scaler.transform(features[self.test_unseen_loc])).float()
        self.test_unseen_features.mul_(1/self.train_features.max())
        self.test_unseen_labels = torch.tensor(labels[self.test_unseen_loc]).long() 
    
        self.seen_classes = torch.tensor(np.unique(self.train_labels.numpy()))
        self.unseen_classes = torch.tensor(np.unique(self.test_unseen_labels.numpy()))
        self.n_train_features = self.train_features.size()[0]
        self.n_train_classes = self.seen_classes.size(0)
        self.n_test_classes = self.unseen_classes.size(0)
        self.train_classes = self.seen_classes.clone()
        self.all_classes = torch.arange(0, self.n_train_classes+self.n_test_classes).long()
    
    def _load_files_hdf5(self, opt):
        file_cls = open(opt.data_path + "/" + opt.dataset + "/" + "allclasses.txt")
        file_cls = file_cls.readlines()
        self.class_names = [c.strip() for c in file_cls]

        # read image feature
        hd5_file = h5py.File(opt.data_path + "/" + opt.dataset + "/" + opt.feat_type + ".hdf5", 'r')
        features = hd5_file['features'][()]
        opt.n_feat = len(features[0]) # check this
        labels = hd5_file['labels'][()] 
        hd5_file.close()
        # read splits and attributes
        mat_file = sio.loadmat(opt.data_path + "/" + opt.dataset + "/" + "att_splits.mat")
        self.trainval_loc = mat_file['trainval_loc'].squeeze() - 1
        #train_loc = mat_file['train_loc'].squeeze() - 1
        #val_unseen_loc = mat_file['val_loc'].squeeze() - 1
        self.test_seen_loc = mat_file['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = mat_file['test_unseen_loc'].squeeze() - 1
        if opt.cls_emb_type == 'att':
            self.cls_embs = torch.tensor(mat_file['att'].T).float() 
        else:
            cls_emb_file = h5py.File(opt.data_path + "/" + opt.dataset + "/"+ opt.cls_emb_type + '.hdf5', 'r')
            self.cls_embs = torch.tensor(cls_emb_file['cls_embedding'].value).float()
        opt.n_cls_emb = len(self.cls_embs[0]) 
        
        scaler = preprocessing.MinMaxScaler()  
        scaler.fit(features[self.trainval_loc])
        self.train_features = torch.tensor(scaler.transform(features[self.trainval_loc])).float()
        self.train_features.mul_(1/self.train_features.max())
        self.train_labels = torch.tensor(labels[self.trainval_loc]).long().squeeze()
        self.test_seen_features = torch.tensor(scaler.transform(features[self.test_seen_loc])).float() 
        self.test_seen_features.mul_(1/self.train_features.max())
        self.test_seen_labels = torch.tensor(labels[self.test_seen_loc]).long().squeeze()
        self.test_unseen_features = torch.tensor(scaler.transform(features[self.test_unseen_loc])).float()
        self.test_unseen_features.mul_(1/self.train_features.max())
        self.test_unseen_labels = torch.tensor(labels[self.test_unseen_loc]).long().squeeze() 
        
        self.seen_classes = torch.tensor(np.unique(self.train_labels.numpy()))
        self.unseen_classes = torch.tensor(np.unique(self.test_unseen_labels.numpy()))
        self.n_train_features = self.train_features.size()[0]
        self.n_train_classes = self.seen_classes.size(0)
        self.n_test_classes = self.unseen_classes.size(0)
        self.train_classes = self.seen_classes.clone()
        self.all_classes = torch.arange(0, self.n_train_classes+self.n_test_classes).long()

    def get_train_dataloader(self, batch_size, use_valset):
        if use_valset == 'no':
            ds = ZSLDataset(self.train_features, self.train_labels, self.cls_embs, self.seen_classes)
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
        else:
            pass    

    def get_random_batch(self, batch_size):
        idx = torch.randperm(self.n_train_features)[0:batch_size]
        batch_feature = self.train_features[idx]
        batch_label = self.train_labels[idx]
        batch_cls_emb = self.cls_embs[batch_label]
        return batch_feature, batch_label, batch_cls_emb


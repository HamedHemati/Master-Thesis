import os
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import models.polynet as model_polynet
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import h5py
from utils import *
from argparse import ArgumentParser


class Dataset_ZSL(Dataset):
    def __init__(self, image_files, model):
        self.image_files = image_files
        image_size = 224
        if model == 'polynet':
        	image_size = 331
        self.trans = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), ])

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        return self.trans(image)

    def __len__(self):
        return len(self.image_files)


class FeatureExtractor(object):
    def __init__(self, args):
        self.args = args
        self.cuda = torch.cuda.is_available()
        self.n_feat = 2048

        # read class list files
        #self.trainval_classes = read_list_from_file(join(self.args.ds_path, 'trainvalclasses.txt'))
        #self.test_classes = read_list_from_file(join(self.args.ds_path, 'testclasses.txt'))

        # set paths for images of trainval and test classes
        #self.ds_images_path = join(self.args.ds_path, 'images')

        # load model
        self.model = None
        self.load_model_()
        if self.cuda:
            self.model = self.model.cuda()
            print("CUDA is available")
        print("Intialized the feature extractor.")

    def load_model_(self):
        if self.args.model == 'resnet-101':
            self.n_feat = 2048
            resnet_101 = models.resnet101(pretrained=False)
            resnet_101.load_state_dict(torch.load(self.args.model_checkpoint_path))
            self.model = nn.Sequential(*list(resnet_101.children())[:-1])
            print("Loaded the resnet-101")
        
        elif self.args.model == 'resnet-18':
            self.n_feat = 512
            resnet_18 = models.resnet18(pretrained=False)
            resnet_18.load_state_dict(torch.load(self.args.model_checkpoint_path))
            self.model = nn.Sequential(*list(resnet_18.children())[:-1])
            print("Loaded the resnet-18")
        
        elif self.args.model == 'alexnet':
            self.n_feat = 4096
            alexnet = models.alexnet(pretrained=False)
            alexnet.load_state_dict(torch.load(self.args.model_checkpoint_path))
            modified_classifier_relu = nn.Sequential(*list(alexnet.classifier.children())[0:-1])
            alexnet.classifier = modified_classifier_relu
            print("Loaded alexnet (with relu)")
            self.model = alexnet

        elif self.args.model == 'vgg16':
            self.n_feat = 4096
            vgg16_bn = models.vgg16(pretrained=False)
            vgg16_bn.load_state_dict(torch.load(self.args.model_checkpoint_path))
            modified_classifier = nn.Sequential(*list(vgg16_bn.classifier.children())[0:-1])
            vgg16_bn.classifier = modified_classifier
            print("Loaded vgg16 (with relu)")
            self.model = vgg16_bn

        elif self.args.model == 'vgg16-bn':
            self.n_feat = 4096
            vgg16_bn = models.vgg16_bn(pretrained=False)
            vgg16_bn.load_state_dict(torch.load(self.args.model_checkpoint_path))
            modified_classifier = nn.Sequential(*list(vgg16_bn.classifier.children())[0:5])
            vgg16_bn.classifier = modified_classifier
            print("Loaded vgg16-bn (with relu)")
            self.model = vgg16_bn

        elif self.args.model == 'vgg19':
            self.n_feat = 4096        
            vgg19 = models.vgg19(pretrained=False)
            vgg19.load_state_dict(torch.load(self.args.model_checkpoint_path))
            modified_classifier = nn.Sequential(*list(vgg19.classifier.children())[0:2])
            vgg19.classifier = modified_classifier
            print("Loaded vgg19 (with relu)")
            self.model = vgg19

        elif self.args.model == 'vgg19-bn':
            self.n_feat = 4096        
            vgg19_bn = models.vgg19_bn(pretrained=False)
            vgg19_bn.load_state_dict(torch.load(self.args.model_checkpoint_path))
            modified_classifier = nn.Sequential(*list(vgg19_bn.classifier.children())[0:2])
            vgg19_bn.classifier = modified_classifier
            print("Loaded vgg19_bn (with relu)")
            self.model = vgg19_bn 

        elif self.args.model == 'polynet':
            self.n_feat = 2048
            polynet = model_polynet.PolyNet()
            polynet.load_state_dict(torch.load(self.args.model_checkpoint_path))
            polynet = nn.Sequential(*list(polynet.children())[:-2])
            print("Loaded polynet (with relu)")
            self.model = polynet

        self.model.eval()

    def compute_feature_(self, image_files):
        ds = Dataset_ZSL(image_files, self.args.model)
        data_loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=False,
                                 num_workers=self.args.num_workers)
        features = np.zeros(self.n_feat)

        for i, feats in enumerate(data_loader):
            batch_size = feats.size(0)
            if self.cuda:
                feats = feats.cuda()   
            out = self.model(feats)
            features = np.vstack((features, out.cpu().data.resize_(batch_size, self.n_feat).numpy()))
            print("%.2f%% is finished" % ((float(i)/len(data_loader))*100))

        features = np.delete(features, 0, axis=0)
        return features

    def extract_features_xlsa17(self):
        xlsa17_data = load_from_xlsa17(self.args.xlsa17_path, self.args.ds_name)
        string_dt = h5py.special_dtype(vlen=str)
        # Splits -------------------------
        with h5py.File(join(self.args.save_path, 'att_splits.hd5'), 'w') as att_splits:
            att_splits.create_dataset('att', data=xlsa17_data['att'])
            allclasses_names = np.array(xlsa17_data['allclasses_names'])
            att_splits.create_dataset('allclasses_names', (len(allclasses_names),), data=allclasses_names.astype('S'),
                                      dtype=string_dt)
            att_splits.create_dataset('test_seen_loc', data=xlsa17_data['test_seen_loc'])
            att_splits.create_dataset('test_unseen_loc', data=xlsa17_data['test_unseen_loc'])
            att_splits.create_dataset('train_loc', data=xlsa17_data['train_loc'])
            att_splits.create_dataset('trainval_loc', data=xlsa17_data['trainval_loc'])
            att_splits.create_dataset('val_loc', data=xlsa17_data['val_loc'])

        # Features -------------------------
        image_path = join(self.args.ds_path, 'images')
        image_files = [join(image_path, x) for x in xlsa17_data['image_files']]
        features = self.compute_feature_(image_files)
        with h5py.File(join(self.args.save_path, 'features.hd5'), 'w') as feat_file:
            feat_file.create_dataset('features', data=features)
            image_files = np.array([[f] for f in xlsa17_data['image_files']])
            feat_file.create_dataset('image_files', (len(image_files),), data=image_files.astype('S'), dtype=string_dt)
            feat_file.create_dataset('labels', data=xlsa17_data['labels'])

    def extract_features2(self, mode):
        """Computes features for every images per class, and stores them into two separate files:
           trainval.hd5: contains features and file names for images of trainval classes
           test.hd5: contains features and file names for images of test classes
        """
        if mode == 'trainval':
            hd5_filename = 'trainval.hd5'
            classes = self.trainval_classes
        elif mode == 'test':
            hd5_filename = 'test.hd5'
            classes = self.test_classes

        # create train file
        print("Train File" + '-'*20)
        with h5py.File(join(self.args.save_path, hd5_filename), 'w') as trainval_file:
            total_cls = len(self.trainval_classes)

            features = np.zeros(self.n_feat)
            file_names = []
            labels = []
            class_names = []

            itr_cls = 0
            for c in classes:
                print('Extracting features for class {}'.format(c))
                class_dir = join(self.ds_images_path, c)
                imgs = os.listdir(class_dir)
                cls_images = [join(class_dir, img) for img in imgs]

                class_feats = self.compute_feature_(cls_images, itr_cls, total_cls-1)

                features = np.vstack((features, class_feats))
                file_names.extend(imgs)
                labels.extend([[itr_cls]] * len(cls_images))
                class_names.append([c])
                itr_cls += 1

            features = np.delete(features, 0, axis=0)
            trainval_file.create_dataset('features', data=features)

            file_names = np.array(file_names).reshape(1,-1).T
            string_dt = h5py.special_dtype(vlen=str)
            trainval_file.create_dataset('file_names', (len(file_names), ), data=file_names.astype('S'),
                                         dtype=string_dt)

            labels = np.array(labels)
            trainval_file.create_dataset('labels', data=labels)

            class_names = np.array(class_names)
            trainval_file.create_dataset('class_names', (len(class_names),), data=class_names.astype('S'),
                                         dtype=string_dt)


def main(args):
    feature_extractor = FeatureExtractor(args)
    feature_extractor.extract_features_xlsa17()
    #feature_extractor2.extract_features2(mode='trainval')
    #feature_extractor2.extract_features2(mode='test')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str)
    parser.add_argument('--xlsa17-path', type=str)
    parser.add_argument('--ds-name', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--model-checkpoint-path', type=str)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=120)
    args = parser.parse_args()
    main(args)

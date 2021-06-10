import torch
import torch.nn as nn
import torch.optim as optim
from .trainer import Trainer
import sys
sys.path.append('..')
from utils.logger import Logger
from utils.plotter import Plotter
from utils.data_handler import DataHandler
from utils.helper_functions import get_new_labels
from utils.feat_sm_classifier import FeatSoftmaxClassifier
from utils.gan_functions import calc_gradient_penalty
from utils.zsl import ZSL
from models.filmwgan import NetD_FiLMWGAN, NetG_FiLMWGAN
from models.filmwgan import NetD_FiLMWGAN2, NetG_FiLMWGAN2


class FiLMWGANTrainer(Trainer):
    """Class FiLMWGANTrainer
    """
    def __init__(self, opt):
        super(FiLMWGANTrainer, self).__init__(opt)
        # model
        if opt.model == 'FiLMWGAN':
            self.net_d = NetD_FiLMWGAN(opt)
            self.net_g = NetG_FiLMWGAN(opt)
        if opt.model == 'FiLMWGAN2':
            self.net_d = NetD_FiLMWGAN2(opt)
            self.net_g = NetG_FiLMWGAN2(opt)   
        self.logger.create_model_log(self.net_d, self.net_g)
        # criterion and optimizer
        self.criterion_cls = nn.NLLLoss()
        if self.cuda:
            print("CUDA is enabled")
            self.net_d.cuda()
            self.net_g.cuda()
            self.criterion_cls.cuda()
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def run(self):
        """Starts training.
        """
        # train the softmax classifer and freeze it
        self.sm_classifier = FeatSoftmaxClassifier(X=self.data_handler.train_features, Y=get_new_labels(self.data_handler.train_labels, self.data_handler.seen_classes),
                                                   n_feat=self.opt.n_feat, n_cls=self.data_handler.seen_classes.size(0), n_epochs=self.opt.n_cls_epochs)
        self.sm_classifier.fit()
        if self.opt.use_cent_loss == 'yes':
	        self._calc_class_centroid()
        # freeze the classifier during the optimization
        for p in self.sm_classifier.model.parameters(): # set requires_grad to False
            p.requires_grad = False
        
        # histograms
        self.train_hists = {'loss_net_d': [], 'loss_net_g': [],  'mean_d_x': [],
                            'mean_d_z1': [], 'mean_d_z2': [], 'wasserstein_dist': [], 'x': []}
        plotter = Plotter(self.opt.outputs_path)
        for epoch in range(self.opt.n_epochs):
            self._train_epoch(epoch)
            if self.opt.eval_zsl == 'yes':
                self._eval_epoch_zsl(epoch)
            if self.opt.eval_gzsl == 'yes':
                self._eval_epoch_gzsl(epoch)
            if self.opt.save_every != 0:
                self._save_checkpoints(epoch)
            plotter.plot_gan_train(self.train_hists, True)
            plotter.plot_gan_eval(self.test_hists, True)
            plotter.plot_gan_eval(self.test_hists, False)
    
    def _get_random_batch(self):
        batch_feature, batch_label, batch_att = self.data_handler.get_random_batch(self.opt.batch_size)
        return batch_feature, get_new_labels(batch_label, self.data_handler.seen_classes), batch_att

    def _train_epoch(self, epoch):
        ############################ Train Epoch
        batch_size = self.opt.batch_size
        for iter_tr in range(0, self.data_handler.n_train_features, batch_size):
            ############################ Update Discriminator
            # unfreeze Discriminator    
            for p in self.net_d.parameters():
                p.requires_grad = True 
            # train net_d for n_iter_d iterations    
            for iter_d in range(self.opt.n_iter_d):
                self.net_d.zero_grad()
                feats, labels, cls_emb = self._get_random_batch()
                if self.cuda:
                    feats, labels, cls_emb = feats.cuda(), labels.cuda(), cls_emb.cuda()
                # train with real samples
                out_d_real = self.net_d(feats, cls_emb)
                mean_d_real = out_d_real.mean()
                # train with fake samples
                noise = torch.randn(batch_size, self.opt.n_z)
                if self.cuda:
                    noise = noise.cuda()
                fake = self.net_g(noise, cls_emb)
                out_d_fake = self.net_d(fake.detach(), cls_emb)
                mean_d_fake = out_d_fake.mean()
                # calculate gradient penalty
                gradient_penalty = calc_gradient_penalty(self.net_d, self.opt, feats, fake.detach(), cls_emb, batch_size, self.cuda)
                # compute total loss wasserstein distance
                wasserstein_dist = mean_d_real.item() - mean_d_fake.item()
                loss_d = mean_d_fake - mean_d_real + gradient_penalty
                loss_d.backward()
                self.optimizer_d.step()
            ############################ Update Generator
            # freeze Discirminator
            for p in self.net_d.parameters(): 
                p.requires_grad = False 
            # train net_g
            self.net_g.zero_grad()
            noise = torch.randn(batch_size, self.opt.n_z)
            centroid_loss = torch.tensor(0.0)
            if self.cuda:
                noise = noise.cuda()
                centroid_loss = centroid_loss.cuda()
            fake = self.net_g(noise, cls_emb)
            out_g_fake = self.net_d(fake, cls_emb)
            mean_g_fake = out_g_fake.mean()
            loss_mean_g = -mean_g_fake
            # classification loss
            loss_cls_g = self.criterion_cls(self.sm_classifier.model(fake), labels)
            # centroid loss:
            if self.opt.use_cent_loss == 'yes':
	            for i in range(self.data_handler.n_train_classes):
	            	idx_cls_feats = (labels==i).data.nonzero().squeeze()
	            	if idx_cls_feats.numel() != 0:
	            		fake_feats_cls = fake[idx_cls_feats, :]
	            		centroid_loss += (fake_feats_cls.mean(dim=0) - self.cls_centroid[i]).pow(2).sum().sqrt()
	            	centroid_loss *= (1.0 / self.data_handler.n_train_classes)
            # net_g total loss
            loss_g =  loss_mean_g + (self.opt.cls_weight * loss_cls_g) + (self.opt.lambda_centl * centroid_loss) 
            loss_g.backward()
            self.optimizer_g.step()
            # print log
            if iter_tr % 5 == 0:
                print("========== Epoch %d - %.2f%%" %(epoch, (float(iter_tr)/self.data_handler.n_train_features) * 100))
                log_msg = "mean(x): %.2f, mean(g(z)): %.2f, mean(g(z)): %.2f, wass_dist: %.2f, loss_cls: %.2f, gp:%.2f, cent_loss: %.4f"%(
                           mean_d_real.item(), mean_d_fake.item(), mean_g_fake.item(), wasserstein_dist, loss_cls_g.item(), gradient_penalty.item(), centroid_loss.item())
                print(log_msg)
                print("\n")
            # update hists
            self.train_hists['loss_net_d'].append(loss_d.item())
            self.train_hists['loss_net_g'].append(loss_g.item())
            self.train_hists['mean_d_x'].append(mean_d_real.item())
            self.train_hists['mean_d_z1'].append(mean_d_fake.item())
            self.train_hists['mean_d_z2'].append(mean_g_fake.item())
            self.train_hists['wasserstein_dist'].append(wasserstein_dist)
            self.train_hists['x'].append(epoch + iter_tr/float(self.data_handler.n_train_features))

    def _calc_class_centroid(self):
    	self.cls_centroid = torch.zeros(self.data_handler.n_train_classes, self.opt.n_feat).float()
    	train_labels = get_new_labels(self.data_handler.train_labels, self.data_handler.seen_classes)
    	for i in range(self.data_handler.n_train_classes):
    		self.cls_centroid[i] = torch.mean(self.data_handler.train_features[train_labels == i], dim=0)
    	if self.cuda:
    		self.cls_centroid = self.cls_centroid.cuda()	
    	print("Class centroids calculated")	
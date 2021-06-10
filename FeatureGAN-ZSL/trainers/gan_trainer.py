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
from models.gan import NetD_GAN, NetG_GAN


class GANTrainer(Trainer):
    """Class GANTrainer
    """
    def __init__(self, opt):
        super(GANTrainer, self).__init__(opt)
        
        self.train_dataloader = self.data_handler.get_train_dataloader(self.opt.batch_size, self.opt.use_valset)
        # model
        if opt.model == 'GAN':
            self.net_d = NetD_GAN(opt)
            self.net_g = NetG_GAN(opt)   
        self.logger.create_model_log(self.net_d, self.net_g)
        # criterion and optimizer
        self.criterion_adv = nn.BCELoss()
        self.criterion_cls = nn.NLLLoss()
        if self.cuda:
            print("CUDA is enabled")
            self.net_d.cuda()
            self.net_g.cuda()
            self.criterion_adv.cuda()
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

        # freeze the classifier during the optimization
        for p in self.sm_classifier.model.parameters(): # set requires_grad to False
            p.requires_grad = False
        
        # histograms
        self.train_hists = {'loss_net_d': [], 'loss_net_g': [],  'mean_d_x': [],
                            'mean_d_z1': [], 'mean_d_z2': [], 'x': []}
        plotter = Plotter(self.opt.outputs_path)
        for epoch in range(self.opt.n_epochs):
            self._train_epoch(epoch)
            if self.opt.eval_zsl == 'yes':
                self._eval_epoch_zsl(epoch)
            if self.opt.eval_gzsl == 'yes':
                self._eval_epoch_gzsl(epoch)
            if self.opt.save_every != 0:
                self._save_checkpoints(epoch)
            plotter.plot_gan_train(self.train_hists, False)
            plotter.plot_gan_eval(self.test_hists, True)
            plotter.plot_gan_eval(self.test_hists, False)
    
    def _train_epoch(self, epoch):
        ############################ Train Epoch
        for iter_tr, (feats, labels, cls_emb) in enumerate(self.train_dataloader): 
            ############################ Update Discriminator
            # unfreeze Discriminator    
            for p in self.net_d.parameters():
                p.requires_grad = True 
            
            # set gradients of the discrimiantor to zero   
            self.net_d.zero_grad()
            lbl_fake = torch.zeros(feats.size(0), 1)
            lbl_real = torch.ones(feats.size(0), 1)
            if self.cuda:
                feats, labels, cls_emb = feats.cuda(), labels.cuda(), cls_emb.cuda()
                lbl_real, lbl_fake = lbl_real.cuda(), lbl_fake.cuda()
            
            # train with real samples
            out_d_real = self.net_d(feats, cls_emb)
            mean_d_real = out_d_real.mean()
            
            # train with fake samples
            noise = torch.randn(feats.size(0), self.opt.n_z)
            if self.cuda:
                noise = noise.cuda()
            fake = self.net_g(noise, cls_emb)
            out_d_fake = self.net_d(fake.detach(), cls_emb)
            mean_d_fake = out_d_fake.mean()

            # compute total loss
            loss_d = 0.5*(self.criterion_adv(out_d_real, lbl_real) + self.criterion_adv(out_d_fake, lbl_fake))
            loss_d.backward()
            self.optimizer_d.step()

            ############################ Update Generator
            # freeze Discirminator
            for p in self.net_d.parameters(): 
                p.requires_grad = False 
            
            # train net_g
            self.net_g.zero_grad()
            noise = torch.randn(feats.size(0), self.opt.n_z)
            if self.cuda:
                noise = noise.cuda()
            fake = self.net_g(noise, cls_emb)
            out_g_fake = self.net_d(fake, cls_emb)
            mean_g_fake = out_g_fake.mean()

            # classification loss
            loss_cls_g = self.criterion_cls(self.sm_classifier.model(fake), labels)
            loss_g =  self.criterion_adv(out_g_fake, lbl_real) + (self.opt.cls_weight*loss_cls_g) 
            
            # net_g total loss
            loss_g.backward()
            self.optimizer_g.step()
            
            # print log
            if iter_tr % 10 == 0:
                print("========== Epoch %d - %.2f%%" %(epoch, (float(iter_tr)/len(self.train_dataloader)) * 100))
                log_msg = "mean(x): %.2f, mean(g(z)): %.2f, mean(g(z)): %.2f, loss_cls: %.2f" %(
                           mean_d_real.item(), mean_d_fake.item(), mean_g_fake.item(), loss_cls_g.item())
                print(log_msg)
                print("\n")
            
            # update hists
            self.train_hists['loss_net_d'].append(loss_d.item())
            self.train_hists['loss_net_g'].append(loss_g.item())
            self.train_hists['mean_d_x'].append(mean_d_real.item())
            self.train_hists['mean_d_z1'].append(mean_d_fake.item())
            self.train_hists['mean_d_z2'].append(mean_g_fake.item())
            self.train_hists['x'].append(epoch + iter_tr/float(self.data_handler.n_train_features))
        
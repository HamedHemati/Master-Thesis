import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.logger import Logger
from utils.plotter import Plotter
from utils.data_handler import DataHandler
from utils.helper_functions import init_env, get_new_labels
from utils.feat_sm_classifier import FeatSoftmaxClassifier
from utils.gan_functions import generate_synthetic_features, calc_gradient_penalty
from utils.zsl import ZSL
import os


class Trainer(object):
    """Class ZSWGANTrainer
    """
    def __init__(self, opt):
        # set manual seeds 
        init_env(opt.random_seed)
        # opt and cuda settings
        self.opt = opt
        self.cuda = self.opt.cuda and torch.cuda.is_available()
        # data loader
        self.data_handler = DataHandler(opt)
        # logger and plotter
        self.logger = Logger(save_dir=self.opt.outputs_path, opt=self.opt)
        self.plotter = Plotter(path=self.opt.outputs_path)
        self.net_d = None
        self.net_g = None
        self.test_hists = {'zsl': [], 'gzsl_seen': [], 'gzsl_unseen': [], 'gzsl_hmean': []}

    def run(self):
        raise NotImplementedError

    def _train_epoch(self):
        raise NotImplementedError

    def _eval_epoch_gzsl(self, epoch):
        print("Evaluating GZSL epoch %d"%(epoch))
        self.net_g.eval()
        if self.opt.use_valset == 'no':
            n_samples = self.opt.n_synth_samples
            if self.opt.model == "AEWGAN":
                syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda, multi_out=True)
            else:    
                syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
            X = torch.cat([self.data_handler.train_features, syn_features], dim=0)
            Y = torch.cat([self.data_handler.train_labels, syn_labels], dim=0)
            n_cls = self.data_handler.n_train_classes + self.data_handler.n_test_classes
            gzsl = ZSL(X, Y, self.data_handler, n_samples, 25, n_cls, 0.001)
            acc_seen, acc_unseen, acc_h = gzsl.run_gzsl()
            print("-"*30)
            print('Acc. Unseen: %.4f%%\nAcc. Seen: %.4f%%\nAcc. H: %.4f%%\n\n' % (acc_unseen, acc_seen, acc_h))
            self.test_hists['gzsl_seen'].append(acc_seen)
            self.test_hists['gzsl_unseen'].append(acc_unseen)
            self.test_hists['gzsl_hmean'].append(acc_h)
        if len(self.test_hists['gzsl_hmean'])>=2:
            if self.test_hists['gzsl_hmean'][-1] >= max(self.test_hists['gzsl_hmean'][:-1]):
                self._save_checkpoints_best("gzsl_best")    
        self.net_g.train()

    def _eval_epoch_zsl(self, epoch):
        print("Evaluating ZSL epoch %d"%(epoch))
        self.net_g.eval()
        if self.opt.use_valset == 'no':
            n_samples = self.opt.n_synth_samples
            if self.opt.model == "AEWGAN":
                syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda, multi_out=True)
            else:     
                syn_features, syn_labels = generate_synthetic_features(self.net_g, self.opt, self.data_handler.unseen_classes, self.data_handler.cls_embs, n_samples, self.cuda)
            n_cls = self.data_handler.n_test_classes
            Y = get_new_labels(syn_labels, self.data_handler.unseen_classes)
            zsl = ZSL(syn_features, Y, self.data_handler, n_samples, 25, n_cls, 0.001)
            acc_seen_zsl = zsl.run_zsl()
            self.test_hists['zsl'].append(acc_seen_zsl)
            print("-"*30)
            print('Acc. ZSL: %.4f%%\n' % (acc_seen_zsl))
        if len(self.test_hists['zsl']) >= 2:
            if self.test_hists['zsl'][-1] >= max(self.test_hists['zsl'][:-1]):
                self._save_checkpoints_best("zsl_best")    
        self.net_g.train()
            
    def _save_checkpoints(self, epoch, save_net_d=False, save_net_g=True):
        """Saves the checkpoints for a given epoch.
        """ 
        if epoch % self.opt.save_every == 0:
            if save_net_d:
                name_netd = "net_d_checkpoints/netd_epoch_" + str(epoch) + ".pth"
                torch.save(self.net_d.state_dict(), os.path.join(self.opt.outputs_path, name_netd))
            if save_net_g:
                name_netg = "net_g_checkpoints/netg_epoch_" + str(epoch) + ".pth"
                torch.save(self.net_g.state_dict(), os.path.join(self.opt.outputs_path, name_netg))
            print("Checkpoints for epoch %d saved successfully" % epoch)                

    def _save_checkpoints_best(self, name):
        name_netg = "net_g_checkpoints/" + name + ".pth"
        torch.save(self.net_g.state_dict(), os.path.join(self.opt.outputs_path, name_netg))